# save_two_rgb_from_openvla_tensor.py
import os
import torch
import numpy as np
from PIL import Image


# === 修改为你的路径 ===
TENSOR_PATH = "energy_vis/pixel_values.pt"
CKPT_DIR = "/work1/aiginternal/yuhang/openvla-oft-yhs/ckpoints/openvla-7b+libero_4_task_suites_no_noops+b3+lr-0.0005+lora-r32+dropout-0.0--image_aug--energy_finetuned--200000_chkpt"
OUT_DIR = "energy_vis/recovered_rgb"
os.makedirs(OUT_DIR, exist_ok=True)

# 1) 读取像素张量 [B, C, H, W]
x = torch.load(TENSOR_PATH, map_location="cpu")
if x.ndim == 4 and x.shape[0] == 1:
    x = x.squeeze(0)                # [C, H, W]
x = x.to(torch.float32)
C, H, W = x.shape

# 2) 判断是否 fused（每张图 6 通道）或单骨干（每张 3 通道）
channels_per_image = 6 if C % 6 == 0 else 3
num_images = C // channels_per_image

# 3) 优先从 checkpoint 读取精确的 mean/std（PrismaticImageProcessor 会带上这些）
means, stds = None, None
processor = None
try:
    from transformers import AutoProcessor
    processor = AutoProcessor.from_pretrained(CKPT_DIR, trust_remote_code=True)
    ip = processor.image_processor
    if hasattr(ip, "means") and hasattr(ip, "stds") and ip.means and ip.stds:
        # 形如 [(m_r,m_g,m_b), (m_r,m_g,m_b)]，顺序与 SigLIP/DINOv2 对应
        means, stds = ip.means, ip.stds
    else:
        mm = getattr(ip, "image_mean", [0.5, 0.5, 0.5])
        ss = getattr(ip, "image_std",  [0.5, 0.5, 0.5])
        means, stds = [tuple(mm)], [tuple(ss)]
except Exception:
    # 回退：SigLIP 常用 mean/std=0.5；DINOv2 常用 ImageNet mean/std
    means = [(0.5, 0.5, 0.5), (0.485, 0.456, 0.406)]
    stds  = [(0.5, 0.5, 0.5), (0.229, 0.224, 0.225)]

def unnormalize_rgb(t3chw, mean, std, fallback_minmax=False):
    """t3chw: [3,H,W] 归一化后的张量 -> PIL RGB"""
    if fallback_minmax:
        # 兜底可视化（不依赖 mean/std）
        y = t3chw.clone()
        y = (y - y.amin(dim=(1,2), keepdim=True)) / (y.amax(dim=(1,2), keepdim=True) - y.amin(dim=(1,2), keepdim=True) + 1e-8)
    else:
        m = torch.tensor(mean).view(3,1,1)
        s = torch.tensor(std ).view(3,1,1)
        y = t3chw * s + m  # inverse of (x-mean)/std
        y = y.clamp(0, 1)
    y = (y.permute(1,2,0).numpy() * 255.0).astype(np.uint8)
    return Image.fromarray(y, mode="RGB")

for i in range(num_images):
    block = x[i*channels_per_image:(i+1)*channels_per_image]  # [6,H,W] 或 [3,H,W]
    if channels_per_image == 6:
        # 前3通道：SigLIP视图；后3通道：DINOv2视图
        siglip_rgb = block[0:3]
        dinov2_rgb = block[3:6]
        try:
            img_siglip = unnormalize_rgb(siglip_rgb, means[0], stds[0])
            img_dino   = unnormalize_rgb(dinov2_rgb, means[min(1, len(means)-1)], stds[min(1, len(stds)-1)])
        except Exception:
            img_siglip = unnormalize_rgb(siglip_rgb, None, None, fallback_minmax=True)
            img_dino   = unnormalize_rgb(dinov2_rgb, None, None, fallback_minmax=True)
        img_siglip.save(os.path.join(OUT_DIR, f"img{i}_siglip.png"))
        img_dino.save(  os.path.join(OUT_DIR, f"img{i}_dinov2.png"))
        # 若你只想要“每张”的单文件，可把 SigLIP 版本视作该张图的 RGB：
        img_siglip.save(os.path.join(OUT_DIR, f"img{i}.png"))
    else:
        # 非 fused：每张 3 通道
        try:
            img = unnormalize_rgb(block, means[0], stds[0])
        except Exception:
            img = unnormalize_rgb(block, None, None, fallback_minmax=True)
        img.save(os.path.join(OUT_DIR, f"img{i}.png"))

print(f"Done. Saved to: {OUT_DIR}")




