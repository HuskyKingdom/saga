import torch
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
from energy_model.model import EnergyModel

def gaussian_perturbation_energy_landscape():
    """高斯扰动参数的energy landscape可视化"""
    
    # 加载数据
    device = "cuda"
    CKPT_DIR = "/work1/aiginternal/yuhang/openvla-oft-yhs/ckpoints/openvla-7b-oft-finetuned-libero-spatial-object-goal-10+libero_4_task_suites_no_noops+b24+lr-0.0005+lora-r32+dropout-0.0--image_aug--energy_freeze--100000_chkpt"
    
    energy_model = EnergyModel(4096, 7).to(device)
    energy_model.load_state_dict(torch.load(CKPT_DIR + "/energy_model--100000_checkpoint.pt"))
    energy_model.eval()
    
    context_hidden = torch.load("energy_vis/context_hidden_ts1.pt", map_location=device)
    expert_action = torch.tensor(
        [[[ 1.0000, -0.1396, -0.4590,  0.0153,  0.1172,  0.3965,  0.0000],
         [ 1.0000, -0.0977, -0.4961,  0.0153,  0.1396,  0.4082,  0.0000],
         [ 1.0000,  0.1494, -0.4980, -0.3613,  0.1396,  0.3828,  0.0000],
         [ 1.0000,  0.3418, -0.4277, -0.6758,  0.1396,  0.3047,  0.0000],
         [ 1.0000,  0.3379, -0.3145, -0.8672,  0.1396,  0.2207,  0.0000],
         [ 0.9922,  0.2441, -0.2354, -0.9297,  0.1289,  0.1494,  0.0000],
         [ 0.8711,  0.0454, -0.1660, -0.8477,  0.0781,  0.1167,  0.0000],
         [ 0.7539, -0.0977, -0.0574, -0.8281,  0.0781,  0.1167,  0.0000]]]
    ).to(device)
    
    # 扰动参数网格
    noise_means = np.linspace(-0.5, 0.5, 20)      # x轴: 噪声均值
    noise_stds = np.linspace(0.01, 1.0, 20)       # y轴: 噪声标准差
    
    energy_surface = np.zeros((len(noise_means), len(noise_stds)))
    
    print("Computing energy landscape...")
    for i, noise_mean in enumerate(noise_means):
        for j, noise_std in enumerate(noise_stds):
            # 生成扰动动作
            noise = torch.randn_like(expert_action) * noise_std + noise_mean
            perturbed_action = expert_action + noise
            
            # 计算energy（多次采样取平均，减少随机性）
            energies = []
            for _ in range(5):
                noise_sample = torch.randn_like(expert_action) * noise_std + noise_mean
                action_sample = expert_action + noise_sample
                with torch.no_grad():
                    energy = energy_model(context_hidden, action_sample)
                    energies.append(energy.cpu().item())
            
            energy_surface[i, j] = np.mean(energies)
    
    # 3D可视化
    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection='3d')
    
    X, Y = np.meshgrid(noise_means, noise_stds)
    surf = ax.plot_surface(X, Y, energy_surface.T, cmap='viridis', alpha=0.8)
    
    # 标记expert action点
    expert_energy = energy_model(context_hidden, expert_action).cpu().item()
    ax.scatter([0], [0], [expert_energy], color='red', s=100, label='Expert Action')
    
    ax.set_xlabel('Noise Mean')
    ax.set_ylabel('Noise Std')
    ax.set_zlabel('Energy')
    ax.set_title('Energy Landscape: Gaussian Perturbation Parameters')
    plt.colorbar(surf)
    plt.legend()
    plt.savefig('/work1/aiginternal/yuhang/openvla-oft-yhs/energy_vis/energy_landscape_gaussian_params.png', dpi=300)


gaussian_perturbation_energy_landscape()