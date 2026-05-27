"""
hf_lerobot_dataset.py

PyTorch Dataset for HuggingFace LeRobot-format robot manipulation datasets.
Reads parquet metadata + chunked mp4 videos, preloads frames, normalizes
actions/proprio, and applies RLDSBatchTransform — producing batches compatible
with the OpenVLA-OFT training pipeline.

Dataset format (e.g. christian0420/so101-poker-yellow-task):
  data/chunk-{N:03d}/file-000.parquet   — frame metadata
  videos/observation.images.{cam}/chunk-{N:03d}/file-000.mp4  — video frames

Global frame index i  →  chunk i // VIDEO_CHUNK_SIZE, position i % VIDEO_CHUNK_SIZE.

Extra dependencies (not in pyproject.toml):
    pip install av opencv-python-headless pandas pyarrow
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import pandas as pd
import torch
from huggingface_hub import snapshot_download
from torch.utils.data import Dataset

from prismatic.vla.constants import ACTION_PROPRIO_NORMALIZATION_TYPE, NormalizationType
from prismatic.vla.datasets.datasets import RLDSBatchTransform

VIDEO_CHUNK_SIZE = 1000  # LeRobot default: 1000 frames per video file


def _decode_video(video_path: str, arr: np.ndarray, start_idx: int, res: int) -> int:
    """
    Decode all frames from a video file into arr[start_idx:], resized to res×res RGB.
    Tries PyAV first (supports AV1/H.265/H.264); falls back to OpenCV.
    Returns the next free index (start_idx + number of frames decoded).
    """
    try:
        import av  # pip install av
        idx = start_idx
        with av.open(video_path) as container:
            for frame in container.decode(video=0):
                img = frame.to_ndarray(format="rgb24")
                img = cv2.resize(img, (res, res), interpolation=cv2.INTER_AREA)
                arr[idx] = img
                idx += 1
        return idx
    except ImportError:
        pass

    # OpenCV fallback (does NOT support AV1 in most builds)
    cap = cv2.VideoCapture(video_path)
    idx = start_idx
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, (res, res), interpolation=cv2.INTER_AREA)
        arr[idx] = frame
        idx += 1
    cap.release()
    return idx


class LeRobotHFDataset(Dataset):
    """
    Dataset for HuggingFace LeRobot-format demonstrations.

    On first use, decodes all video frames into numpy arrays cached on disk
    (at preload_resolution × preload_resolution) so subsequent runs are fast.
    Actions and proprio are normalized to [-1, 1] using the dataset's own
    statistics before being passed to batch_transform.

    Camera names are auto-detected from videos/observation.images.* directories.
    The first camera becomes image_primary; the second (if present) becomes image_wrist.
    """

    def __init__(
        self,
        repo_id: str,
        batch_transform: RLDSBatchTransform,
        chunk_size: int = 10,
        task_instruction: str = "pick up the yellow poker chip",
        cache_dir: Optional[str] = None,
        preload_resolution: int = 256,
        train: bool = True,
        val_split: float = 0.1,
    ):
        self.repo_id = repo_id
        self.batch_transform = batch_transform
        self.chunk_size = chunk_size
        self.task_instruction_bytes = task_instruction.encode()
        self.dataset_name = repo_id.split("/")[-1].replace("-", "_")
        self.preload_resolution = preload_resolution

        print(f"Locating dataset {repo_id} ...")
        self.dataset_dir = Path(
            snapshot_download(repo_id=repo_id, repo_type="dataset", cache_dir=cache_dir)
        )
        print(f"  dataset dir: {self.dataset_dir}")

        self._load_metadata()
        self._detect_cameras()
        self._preload_frames()
        self.dataset_statistics = self._compute_statistics()
        self._setup_normalization()
        self._build_samples(train, val_split)

    # ------------------------------------------------------------------
    # Initialization helpers
    # ------------------------------------------------------------------

    def _load_metadata(self) -> None:
        """Load all parquet shards, sort by global index, and build fast numpy arrays."""
        data_dir = self.dataset_dir / "data"
        parquet_files = sorted(data_dir.glob("**/*.parquet"))
        if not parquet_files:
            raise RuntimeError(f"No parquet files found in {data_dir}")

        dfs = [pd.read_parquet(f) for f in parquet_files]
        df = pd.concat(dfs).sort_values("index").reset_index(drop=True)

        # Verify global index is a contiguous range 0..N-1 (LeRobot convention)
        expected = list(range(len(df)))
        actual = df["index"].tolist()
        if actual != expected:
            raise ValueError(
                f"Global frame index is not contiguous 0..{len(df)-1}. "
                "Cannot use direct array indexing."
            )

        self.df = df
        self.n_frames = len(df)

        # Pre-extract action and state arrays for O(1) access in __getitem__
        self.all_actions = np.stack(df["action"].tolist()).astype(np.float32)   # (N, action_dim)
        self.all_states = np.stack(df["observation.state"].tolist()).astype(np.float32)  # (N, state_dim)

        n_ep = df["episode_index"].nunique()
        print(f"  metadata: {self.n_frames} frames, {n_ep} episodes")

    def _detect_cameras(self) -> None:
        """Auto-detect camera names from videos/observation.images.* directories."""
        videos_dir = self.dataset_dir / "videos"
        cam_dirs = sorted(
            d for d in videos_dir.iterdir()
            if d.is_dir() and d.name.startswith("observation.images.")
        )
        if not cam_dirs:
            raise RuntimeError(f"No observation.images.* directories found in {videos_dir}")
        self.cameras: List[str] = [d.name.split("observation.images.", 1)[1] for d in cam_dirs]
        print(f"  cameras detected: {self.cameras}")

    def _preload_frames(self) -> None:
        """
        Decode all video frames and store as uint8 numpy arrays keyed by camera name.
        Results are cached to disk so subsequent runs skip decoding.

        Uses an exclusive file lock (fcntl.flock) so that multiple DDP processes
        running simultaneously do not corrupt the cache by writing concurrently.
        The first process to acquire the lock decodes and writes; subsequent
        processes find the cache already built and just load it.
        """
        import fcntl

        res = self.preload_resolution
        cache_paths = {cam: self.dataset_dir / f"_cache_{cam}_{res}.npy" for cam in self.cameras}
        lock_path = self.dataset_dir / f"_frame_cache_{res}.lock"
        expected_shape = (self.n_frames, res, res, 3)

        def _try_load() -> bool:
            if not all(p.exists() for p in cache_paths.values()):
                return False
            try:
                frames = {}
                for cam, path in cache_paths.items():
                    arr = np.load(str(path))
                    assert arr.shape == expected_shape, f"{cam} shape {arr.shape} != {expected_shape}"
                    frames[cam] = arr
                self.all_cam_frames = frames
                return True
            except Exception as e:
                print(f"  cache invalid ({e}), will rebuild ...")
                for p in cache_paths.values():
                    p.unlink(missing_ok=True)
                return False

        # Fast path: if cache is already valid, skip locking entirely
        if _try_load():
            print("  loading cached frames ...")
            return

        # Slow path: acquire exclusive lock, then build (or load if another
        # process already built while we were waiting for the lock)
        with open(lock_path, "w") as lf:
            fcntl.flock(lf, fcntl.LOCK_EX)   # blocks until we own the lock

            if _try_load():   # another process may have built it while we waited
                print("  loading cached frames (built by peer process) ...")
                return

            print(f"  decoding video frames at {res}×{res} (first-run, may take ~1 min) ...")
            frames = {cam: np.zeros(expected_shape, dtype=np.uint8) for cam in self.cameras}

            for cam in self.cameras:
                arr = frames[cam]
                video_root = self.dataset_dir / "videos" / f"observation.images.{cam}"
                video_files = sorted(video_root.glob("**/*.mp4"))
                if not video_files:
                    raise RuntimeError(f"No mp4 files found under {video_root}")
                global_idx = 0
                for vf in video_files:
                    global_idx = _decode_video(str(vf), arr, global_idx, res)
                if global_idx != self.n_frames:
                    raise RuntimeError(
                        f"{cam}: decoded {global_idx} frames but expected {self.n_frames}. "
                        "OpenCV likely cannot decode the video codec (e.g. AV1). Install PyAV: pip install av"
                    )
                print(f"    {cam}: {global_idx} frames decoded")

            # Atomic write via tmp → rename to avoid partial files.
            # Names must end in .npy so numpy does NOT auto-append the extension.
            tmp_paths = {cam: self.dataset_dir / f"_cache_{cam}_{res}_tmp.npy" for cam in self.cameras}
            for cam in self.cameras:
                np.save(str(tmp_paths[cam]), frames[cam])
            for cam in self.cameras:
                tmp_paths[cam].rename(cache_paths[cam])

            self.all_cam_frames = frames
            print("  frames cached to disk.")
        # lock released here

    def _compute_statistics(self) -> Dict:
        """
        Compute per-dim statistics (mean, std, min, max, q01, q99) for
        actions and proprio over the full dataset.

        The returned dict follows the same schema as RLDS dataset_statistics.json
        so it can be saved with save_dataset_statistics() and used during eval.
        """

        def _stats(arr: np.ndarray) -> Dict:
            return {
                "mean": arr.mean(axis=0),
                "std": np.maximum(arr.std(axis=0), 1e-8),
                "min": arr.min(axis=0),
                "max": arr.max(axis=0),
                "q01": np.quantile(arr, 0.01, axis=0),
                "q99": np.quantile(arr, 0.99, axis=0),
                "num_trajectories": int(self.df["episode_index"].nunique()),
                "num_transitions": int(self.n_frames),
            }

        return {
            self.dataset_name: {
                "action": _stats(self.all_actions),
                "proprio": _stats(self.all_states),
            }
        }

    def _setup_normalization(self) -> None:
        """Cache normalization bounds as float32 arrays for fast per-sample use."""
        stats = self.dataset_statistics[self.dataset_name]
        norm = ACTION_PROPRIO_NORMALIZATION_TYPE

        if norm == NormalizationType.BOUNDS_Q99:
            self._act_lo = stats["action"]["q01"].astype(np.float32)
            self._act_hi = stats["action"]["q99"].astype(np.float32)
            self._pro_lo = stats["proprio"]["q01"].astype(np.float32)
            self._pro_hi = stats["proprio"]["q99"].astype(np.float32)
        elif norm == NormalizationType.BOUNDS:
            self._act_lo = stats["action"]["min"].astype(np.float32)
            self._act_hi = stats["action"]["max"].astype(np.float32)
            self._pro_lo = stats["proprio"]["min"].astype(np.float32)
            self._pro_hi = stats["proprio"]["max"].astype(np.float32)
        else:  # NORMAL
            self._act_lo = None  # signal to use z-score
            self._act_mean = stats["action"]["mean"].astype(np.float32)
            self._act_std = stats["action"]["std"].astype(np.float32)
            self._pro_mean = stats["proprio"]["mean"].astype(np.float32)
            self._pro_std = stats["proprio"]["std"].astype(np.float32)

        self._norm_type = norm

    def _build_samples(self, train: bool, val_split: float) -> None:
        """
        Partition episodes into train / val and build a flat list of
        (episode_id, position_within_episode) tuples.
        """
        episode_ids = sorted(self.df["episode_index"].unique().tolist())
        n_val = max(1, round(len(episode_ids) * val_split))
        target_ids = set(episode_ids[:-n_val] if train else episode_ids[-n_val:])

        # Map episode_id → sorted list of global frame indices
        self._ep_global_idxs: Dict[int, List[int]] = {}
        for ep_id, grp in self.df.groupby("episode_index"):
            if ep_id in target_ids:
                self._ep_global_idxs[int(ep_id)] = grp.sort_values("frame_index")["index"].tolist()

        self.samples: List[Tuple[int, int]] = [
            (ep_id, pos)
            for ep_id, idxs in self._ep_global_idxs.items()
            for pos in range(len(idxs))
        ]

        split = "train" if train else "val"
        print(f"  {split} samples: {len(self.samples)} from {len(self._ep_global_idxs)} episodes")

    # ------------------------------------------------------------------
    # Normalization
    # ------------------------------------------------------------------

    def _normalize(self, values: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> np.ndarray:
        scale = np.maximum(hi - lo, 1e-8)
        return np.clip(2.0 * (values - lo) / scale - 1.0, -1.0, 1.0)

    def _norm_action(self, actions: np.ndarray) -> np.ndarray:
        if self._norm_type == NormalizationType.NORMAL:
            return (actions - self._act_mean) / self._act_std
        return self._normalize(actions, self._act_lo, self._act_hi)

    def _norm_proprio(self, state: np.ndarray) -> np.ndarray:
        if self._norm_type == NormalizationType.NORMAL:
            return (state - self._pro_mean) / self._pro_std
        return self._normalize(state, self._pro_lo, self._pro_hi)

    # ------------------------------------------------------------------
    # Dataset interface
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict:
        ep_id, pos = self.samples[idx]
        global_idxs = self._ep_global_idxs[ep_id]
        n_ep = len(global_idxs)

        # Action chunk: [pos, pos+chunk_size), pad last action if near episode end
        end = min(pos + self.chunk_size, n_ep)
        chunk_global = global_idxs[pos:end]
        actions = self.all_actions[chunk_global].copy()  # (actual_len, action_dim)
        if len(actions) < self.chunk_size:
            pad = np.tile(actions[-1:], (self.chunk_size - len(actions), 1))
            actions = np.concatenate([actions, pad], axis=0)

        # Current proprio
        cur_g = global_idxs[pos]
        proprio = self.all_states[cur_g].copy()  # (proprio_dim,)

        # Normalize to [-1, 1]
        actions = self._norm_action(actions)
        proprio = self._norm_proprio(proprio)

        # cameras[0] → image_primary, cameras[1] → image_wrist (if present)
        primary_img = self.all_cam_frames[self.cameras[0]][cur_g]
        observation = {
            "image_primary": primary_img[np.newaxis],  # (1, H, W, 3)
            "proprio": proprio,
        }
        if len(self.cameras) > 1:
            wrist_img = self.all_cam_frames[self.cameras[1]][cur_g]
            observation["image_wrist"] = wrist_img[np.newaxis]  # (1, H, W, 3)

        rlds_batch = {
            "dataset_name": self.dataset_name,
            "action": actions,
            "observation": observation,
            "task": {
                "language_instruction": self.task_instruction_bytes,
            },
        }
        return self.batch_transform(rlds_batch)
