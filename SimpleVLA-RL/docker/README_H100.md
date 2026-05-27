# SAGA RL on NVIDIA H100 — Enroot + Pyxis

This is the CUDA / Enroot port of the AMD ROCm + Apptainer flow used by
`saga_rl_trail.sh` -> `examples/run_saga.sh` (`simplevla-rl-rocm.def`).

## Layout

```
SimpleVLA-RL/
├── docker/
│   ├── Dockerfile.h100          ← CUDA image definition (NGC PyTorch 24.10)
│   ├── build_h100.sh            ← docker build + enroot import + optional scp
│   └── README_H100.md           ← this file
├── examples/
│   └── run_saga_h100.sh         ← SLURM batch script using pyxis
└── saga_rl_trail_h100.sh        ← top-level launcher (counterpart of saga_rl_trail.sh)
```

## End-to-end workflow

### 1. Build the image locally

The build context must be the **openvla-oft-yhs repo root** so the Dockerfile
can `COPY . /workspace/openvla-oft-yhs`.

```bash
cd /home/yuhang/Desktop/oft-workspace/openvla-oft-yhs

# Build docker image + export to enroot squashfs
bash SimpleVLA-RL/docker/build_h100.sh
# Produces:  ./simplevla-rl-cuda-saga.sqsh
```

To also push the .sqsh to the H100 cluster in one shot:

```bash
UPLOAD=1 \
REMOTE_HOST=yuhang@h100-login.example.com \
REMOTE_DIR=/scratch/yuhang/containers \
    bash SimpleVLA-RL/docker/build_h100.sh
```

If `enroot` isn't installed locally, the script will skip the `enroot import`
step. You can run that step on any machine with enroot, or push the docker
image to a registry and `enroot import docker://registry.example.com/...` from
the H100 side.

### 2. Stage runtime data on the H100 cluster

The container is **self-contained for code** — `prismatic`, `verl`, `LIBERO`,
and the SimpleVLA-RL entrypoints are all baked into `/workspace/openvla-oft-yhs`.
You only need to bring three things from the host:

| Host                                | Container                          | Purpose                |
|-------------------------------------|------------------------------------|------------------------|
| `$HOST_SFT_MODEL_DIR`               | `/mnt/sft_models/<basename>` (RO)  | SFT starting checkpoint |
| `$HOST_APD_PLANS_FILE`              | `/mnt/apd_plans.json`     (RO)     | APD plans JSON         |
| `$HOST_CKPT_DIR`                    | `/mnt/ckpt`                        | Where saga writes ckpts |

Typical staging:

```bash
# On the H100 login node (or wherever you have storage):
mkdir -p ~/containers ~/sft_models ~/saga_ckpts
# scp the .sqsh, the SFT checkpoint dir, and APD_plans_scaled.json over
scp simplevla-rl-cuda-saga.sqsh   h100:~/containers/
rsync -avh oft_plus_discrete/      h100:~/sft_models/oft_plus_discrete/
scp APD_plans_scaled.json          h100:~/APD_plans_scaled.json
```

### 3. Submit the job

```bash
ssh h100
cd <wherever the SimpleVLA-RL repo lives>      # only needed if you want HOST_REPO_OVERRIDE

# Edit the four HOST_* paths at the top of saga_rl_trail_h100.sh, then:
bash SimpleVLA-RL/saga_rl_trail_h100.sh
# → sbatch examples/run_saga_h100.sh
```

Add `--partition=...` / `--account=...` to the `#SBATCH` block in
`run_saga_h100.sh` for your cluster — the template doesn't ship with one.

### 4. Iterating on code without rebuilding

If you're hacking on `verl/` or `prismatic/` and don't want to rebuild the
image every time, set `HOST_REPO_OVERRIDE` to bind your host repo over the
baked-in `/workspace/openvla-oft-yhs`:

```bash
HOST_REPO_OVERRIDE=/home/yuhang/openvla-oft-yhs \
    bash SimpleVLA-RL/saga_rl_trail_h100.sh
```

## Mapping from the ROCm flow

| ROCm / Apptainer                   | CUDA / Enroot                          |
|-----------------------------------|---------------------------------------|
| `simplevla-rl-rocm.def`           | `docker/Dockerfile.h100`              |
| `apptainer build *.sif`           | `docker build` + `enroot import`      |
| `*.sif`                           | `*.sqsh`                              |
| `apptainer run --bind ...`        | `srun --container-image / --container-mounts` |
| `HIP_VISIBLE_DEVICES`             | `CUDA_VISIBLE_DEVICES`                |
| `triton-rocm`                     | `triton` (bundled in NGC PyTorch)     |
| `flash-attn` (ROCm fork)          | `flash-attn` (bundled in NGC PyTorch) |
| `/opt/rocm` LD paths              | not needed — NGC handles CUDA libs    |

## Hydra overrides

`run_saga_h100.sh` keeps **all** Hydra overrides identical to `examples/run_saga.sh`.
The only differences are:
- Host-side paths replaced with their in-container equivalents (`/mnt/...`, `/workspace/...`).
- ROCm-only env vars dropped; CUDA equivalents added.

## GPU rendering (LIBERO rollouts on the H100, not the CPU)

The AMD ROCm flow set `MUJOCO_GL=osmesa` because the AMD/container combo
couldn't initialise GPU EGL. On NV H100 we want LIBERO's MuJoCo rollouts to
render on the GPU, so this image defaults to:

```
MUJOCO_GL=egl
PYOPENGL_PLATFORM=egl
NVIDIA_VISIBLE_DEVICES=all
NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics
```

The last var is the load-bearing one: NGC's default is `compute,utility`,
which makes enroot/pyxis skip injecting `libEGL_nvidia.so.0` from the host
driver. MuJoCo will then either error or silently fall back to software
(CPU) rendering, and rollouts get ~10× slower.

**Verify GPU rendering is actually on** (run inside the container):

```bash
python -c "
import mujoco, os
print('MUJOCO_GL =', os.environ.get('MUJOCO_GL'))
m = mujoco.MjModel.from_xml_string('<mujoco><worldbody><body><geom size=\"1\"/></body></worldbody></mujoco>')
d = mujoco.MjData(m)
r = mujoco.Renderer(m, 256, 256)
r.update_scene(d); r.render()
print('Renderer backend OK:', r.__class__.__name__)
"
# In a second shell on the same node:  nvidia-smi  → should show python using a few hundred MB of VRAM
```

If you see `EGL: Failed to get platform display` or `libEGL_nvidia.so.0: not
found`, the driver caps env var didn't make it through. Check that
`NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics` is exported in the
SLURM environment **before** `srun` (the run script does this).

To force CPU rendering for debugging (matches AMD behaviour):

```bash
MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa bash SimpleVLA-RL/saga_rl_trail_h100.sh
```

## Troubleshooting

**`flash-attn` import errors**
Don't `pip install flash-attn` inside `INNER_CMD` — NGC PyTorch already ships
a sm_90-compatible build. Reinstalling can downgrade to a CPU/CUDA-mismatched
wheel.

**NCCL hangs on multi-NIC nodes**
NCCL_DEBUG=WARN is on by default; set `NCCL_DEBUG=INFO` and
`NCCL_SOCKET_IFNAME=...` to match your cluster's preferred interface.

**Read-only `/workspace` complaints**
Pyxis mounts the rootfs read-only by default. The run script writes into
`/tmp/...` (caches) and `/mnt/ckpt/...` (saved checkpoints), both of which are
writable. If a piece of code needs to write inside `/workspace`, use
`HOST_REPO_OVERRIDE` to bind a writable host directory over it.
