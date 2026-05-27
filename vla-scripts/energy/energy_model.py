from typing import List, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

from .layers import FFWRelativeSelfAttentionModule, FFWRelativeCrossAttentionModule
from .position_encodings import PositionalEncoding


class SeqPool(nn.Module):
    def __init__(self, mode="mean"):
        super().__init__()
        assert mode in ["cls", "mean"]
        self.mode = mode

    def forward(self, h):  # h: [B,S,Dh]
        if self.mode == "cls":
            return h[:, 0, :]                       
        else:
            return h.mean(dim=1)
        


class MLPResNetBlock(nn.Module):
    """One MLP ResNet block with a residual connection."""
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.ffn = nn.Sequential(  # feedforward network, similar to the ones in Transformers
            nn.LayerNorm(dim),
            nn.Linear(dim, dim),
            nn.SiLU(),
        )

    def forward(self, x):
        # x: (batch_size, hidden_dim)
        # We follow the module ordering of "Pre-Layer Normalization" feedforward networks in Transformers as
        # described here: https://arxiv.org/pdf/2002.04745.pdf
        identity = x
        x = self.ffn(x)
        x = x + identity
        return x


class MLPResNet(nn.Module):
    """MLP with residual connection blocks."""
    def __init__(self, num_blocks, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.layer_norm1 = nn.LayerNorm(input_dim)
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.act = nn.SiLU()
        self.mlp_resnet_blocks = nn.ModuleList()
        for _ in range(num_blocks):
            self.mlp_resnet_blocks.append(MLPResNetBlock(dim=hidden_dim))
        self.layer_norm2 = nn.LayerNorm(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # x: (batch_size, input_dim)
        x = self.layer_norm1(x)  # shape: (batch_size, input_dim)
        x = self.fc1(x)  # shape: (batch_size, hidden_dim)
        x = self.act(x)  # shape: (batch_size, hidden_dim)
        for block in self.mlp_resnet_blocks:
            x = block(x)  # shape: (batch_size, hidden_dim)
        x = self.layer_norm2(x)  # shape: (batch_size, hidden_dim)
        x = self.fc2(x)  # shape: (batch_size, output_dim)
        return x



# class EnergyModel(nn.Module):
#     """
#     E_phi(s, a):
#     input: hN(s) [B, seq, D_h], a [B, chunk, D_a] 
#     output: energy [B, 1]
#     """
#     def __init__(
#         self,
#         state_dim: int,
#         act_dim: int,
#         hidden: int = 512,
#         n_layers: int = 4,
#         NUM_ACTIONS_CHUNK = 8,
#     ):
#         super().__init__()
#         in_dim = hidden * 3
#         self.action_proj = nn.Linear(act_dim, hidden)
#         self.action_proj_act = nn.SiLU()
        
#         self.state_dim = state_dim
#         self.pool = SeqPool(mode="mean")
#         self.proj_hidden  = nn.Linear(state_dim, hidden)

#         self.model = MLPResNet(
#             num_blocks=n_layers, input_dim=in_dim, hidden_dim=hidden, output_dim=1
#         )


#         # pos emb
#         self.pos_emb = nn.Embedding(NUM_ACTIONS_CHUNK, hidden)


#     def forward(self, hN: torch.Tensor, a: torch.Tensor, reduce="sum", gamma=None) -> torch.Tensor:
#         """
#         hN: [B, S, D_h], a: [B, H,  D_a]
#         return: energy [B, 1]
#         """

#         B, H, Da = a.shape
#         # c = self.pool(hN) # [B, 1]
#         c = hN
#         c = self.proj_hidden(c) # [B, Hd]
#         c = c.unsqueeze(1).expand(B, H, c.shape[-1])  # [B,H,Hd]

#         a = self.action_proj_act(self.action_proj(a)) # [B,H,Hd]

#         # pos emb
#         feats = [c, a]
#         pos_ids = torch.arange(H, device=a.device).unsqueeze(0).expand(B, H)  # [B,H]
#         p = self.pos_emb(pos_ids)                                             # [B,H,Hid]
#         feats.append(p)
#         x = torch.cat(feats, dim=-1)         

#         E_steps = self.model(x)           # [B,H,1]
#         # E_steps = F.softplus(E_steps) # reg
#         E_steps = 0.5 * (E_steps ** 2) + 1e-6

#         if reduce == "sum":
#             if gamma is None:
#                 E = E_steps.sum(dim=1)        # [B,1]
#             else:
#                 w = torch.pow(gamma, torch.arange(H, device=a.device)).view(1,H,1)  # discount
#                 E = (E_steps * w).sum(dim=1)
#         elif reduce == "mean":
#             E = E_steps.mean(dim=1)
#         else:
#             raise ValueError("reduce must be 'sum' or 'mean'")


#         return E, E_steps
    
def assert_finite(x, name):
    if not torch.isfinite(x).all():
        bad = (~torch.isfinite(x)).nonzero(as_tuple=False)[:5]
        raise RuntimeError(f"[NaN] {name} has non-finite at {bad.shape[0]} positions, e.g. {bad[:3].tolist()}")



class EnergyModel(nn.Module):
    """
    E_phi(s, a):
    input: hN(s) [B, seq, D_h], a [B, chunk, D_a] 
    output: energy [B, 1]
    """
    def __init__(
        self,
        state_dim: int,
        act_dim: int,
        hidden: int = 512,
        head: int = 8,
        layers: int = 4,
    ):
        super().__init__()

     
        # self.energy_bc = FFWRelativeCrossAttentionModule(hidden,head,layers)
        self.cross = nn.MultiheadAttention(hidden, head, batch_first=True)

        # pos emb
        self.pe_layer = PositionalEncoding(hidden,0.2)

        self.state_linear = MLPResNet(
            num_blocks=1, input_dim=state_dim, hidden_dim=hidden, output_dim=hidden
        )
        self.action_linear = MLPResNet(
            num_blocks=1, input_dim=act_dim, hidden_dim=hidden, output_dim=hidden
        )
        self.prediction_head = MLPResNet(
            num_blocks=2, input_dim=hidden, hidden_dim=hidden, output_dim=1
        )
        self.pool = SeqPool(mode="mean")


        self.T = 30.0               # temperature for energy range
        
        self.act = nn.Sigmoid() 
        # self.cls_token = nn.Parameter(torch.zeros(1, 1, hidden))
        self.energy_scale = 2.0
        self.energy_offset = 0.1
    

    def forward(self, hN: torch.Tensor, a: torch.Tensor, pad_mask = None, reduce="sum", gamma=None) -> torch.Tensor:
        """
        hN: [B, S, D_h], a: [B, H,  D_a]
        return: energy [B, 1]
        """

        # context_mapped = self.state_linear(hN)  # [B,S,Dh]
        # action_mapped  = self.pe_layer(self.action_linear(a))  # [B,H,Da]
        # cls_tokens = self.cls_token.expand(hN.shape[0], -1, -1)

        # energy_concat = torch.cat([cls_tokens, context_mapped, action_mapped], dim=1).to(hN.dtype)  # [B,S+H+1,Da]

        # energy_features = self.energy_bc(energy_concat.transpose(0,1), diff_ts=None,
        #         query_pos=None, context=None, context_pos=None,pad_mask=pad_mask)[-1].transpose(0,1)  # [B,S+H+1,Da]
        

        # # energy_cls = energy_features[:,0,:].squeeze(1)
        # E = self.prediction_head(energy_cls) # [B, 1]


        hN = hN.float()
        a  = a.float()
        
        assert_finite(hN, "hN")
        assert_finite(a,  "a")
        

        if pad_mask is not None:
            if pad_mask.all(dim=1).any():
                raise RuntimeError("[NaN-risk] some rows key_padding_mask are all True")


    
        # return E
        context_mapped = self.state_linear(hN)  # [B,S,Dh]
        action_mapped  = self.pe_layer(self.action_linear(a))  # [B,H,Da]
        # assert_finite(context_mapped, "context_mapped")
        # assert_finite(action_mapped,  "action_mapped")

        # energy_feat = self.energy_bc(query=action_mapped.transpose(0, 1),
        #     value=context_mapped.transpose(0, 1),
        #     query_pos=None,
        #     value_pos=None,
        #     diff_ts=None)[-1].transpose(0,1) # [B,H,Da]
        
        # energy_feat = energy_feat + self.gate_a * action_mapped 

        # energy = self.pool(energy_feat) # [B,Da]
        # E = self.prediction_head(energy) # [B, 1]


        Z, _ = self.cross(query=action_mapped, key=context_mapped, value=context_mapped, need_weights=False, key_padding_mask=pad_mask)
        # assert_finite(Z, "attn_out")

        energy_feature_step = self.prediction_head(Z)
        # assert_finite(energy_feature_step, "energy_feature_step")


        # raw = self.T * torch.tanh(raw / self.T)
        energy_feature_step = energy_feature_step * 0.5
        E = self.act(energy_feature_step) * self.energy_scale + self.energy_offset
        # assert_finite(E, "E")

        energy_avg = self.pool(E)
        assert_finite(energy_avg, "energy_avg")

        return energy_avg

        





@torch.no_grad()
def one_step_energy_correction_seq(energy_head, h, A_bc, alpha=0.1, clip_frac=0.2,
                                   act_range=None, correct_first_only=False):
    """
    对整块或仅第1步动作做能量梯度校正
    h:  [B,S,Dh], A_bc: [B,H,Da]
    """
    B, H, Da = A_bc.shape
    A = A_bc.detach().clone().requires_grad_(True)      # [B,H,Da]
    E, _ = energy_head(h, A, reduce="sum")              # 标量能量（对整块）
    grad_A = torch.autograd.grad(E.sum(), A)[0]         # [B,H,Da]

    if correct_first_only:
        mask = torch.zeros_like(grad_A); mask[:,0,:] = 1.0
        grad_A = grad_A * mask

    step = alpha * grad_A
    if act_range is not None:
        max_step = clip_frac * act_range.view(1,1,-1).to(step.device)
        step = torch.clamp(step, -max_step, max_step)
    else:
        # 全局范数裁剪
        step_norm = step.flatten(1).norm(dim=-1, keepdim=True) + 1e-6
        base_norm = A_bc.flatten(1).norm(dim=-1, keepdim=True) + 1e-6
        coef = torch.minimum(torch.ones_like(step_norm), (clip_frac*base_norm)/step_norm)
        step = step * coef.view(B,1,1)

    A_ref = A - step
    return A_ref.detach()





# def compute_negative_energy(energy_head, A_star,layer_actions,delta,hidden_N, P_loss, topk=3,kappa=1):

#     B, H, Da = A_star.shape
#     cand_idx, cand_A, cand_dist = [], [], []
    
#     with torch.no_grad():
#         for A_j in layer_actions:
#             dist = torch.norm((A_j - A_star).reshape(B, -1), dim=-1)  # [B]
#             mask = dist > delta
#             if mask.any():
#                 idx = torch.nonzero(mask, as_tuple=False).squeeze(-1)
#                 cand_idx.append(idx)
#                 cand_A.append(A_j[idx])                # [B_sel,H,Da]
#                 cand_dist.append(dist[idx])

#     if len(cand_idx) == 0:
#         return None

#     idx_cat = torch.cat(cand_idx, dim=0)           # [B']
#     A_cat   = torch.cat(cand_A,   dim=0)           # [B',H,Da]
#     d_cat   = torch.cat(cand_dist, dim=0)          # [B']

#     per_i_rows = [[] for _ in range(B)]
#     for row, i in enumerate(idx_cat.tolist()):
#         per_i_rows[i].append(row)

#     keep_rows = []
#     for rows in per_i_rows:
#         if not rows:
#             continue
#         rows_sorted = sorted(rows, key=lambda r: float(d_cat[r]), reverse=True)[:topk]
#         keep_rows.extend(rows_sorted)

#     if len(keep_rows) == 0:
#         return None

#     keep_rows = torch.tensor(keep_rows, dtype=torch.long, device=A_star.device)
#     A_neg = A_cat[keep_rows]                        # [B'',H,Da]
#     idx   = idx_cat[keep_rows]                      # [B'']

#     E_neg, _ = energy_head(hidden_N[idx], A_neg)  # [B'',1]

#     with torch.no_grad():

#         margin = kappa * torch.norm((A_neg - A_star[idx]).reshape(A_neg.shape[0], -1),
#                                     dim=-1, keepdim=True)         # [B'',1]

#     # E_pos detach
#     L_neg = F.relu(margin + P_loss[idx] - E_neg).mean()

#     return L_neg


def add_gaussian_noise(x: torch.Tensor,
                       sigma: float,
                       mu: float = 0.0,
                       clamp: tuple | None = None,   # 例如 (0, 1)
                       per_channel: bool = False,     # True=按通道共享同一噪声
                       generator: torch.Generator | None = None):
    """
    x: tensor in any shape
    sigma: for noise
    mu: for noise
    clamp: optional
    per_channel: 对4D(N,C,H,W)等按通道共享噪声；其他shape按第1维视为通道
    """

    # 解决半精度(bfloat16/float16)在GPU上直采样质量差的问题：先用fp32采样再转回
    noise_dtype = torch.float32
    dev = x.device

    if per_channel:
        if x.ndim >= 2:
            shape = [1, x.shape[1]] + [1] * (x.ndim - 2)
        else:
            shape = [1] * x.ndim
        noise = torch.randn(shape, dtype=noise_dtype, device=dev)
        noise = noise.expand_as(x)
    else:
        noise = torch.randn_like(x, dtype=noise_dtype, device=dev)

    y = x + noise.to(x.dtype) * sigma + torch.as_tensor(mu, dtype=x.dtype, device=dev)

    if clamp is not None:
        y = y.clamp(*clamp)
    return y

# def compute_negative_energy(energy_head, A_star, layer_actions, delta, hidden_N, P_loss, topk=2, kappa=1):

#     A_neg = layer_actions[1]  
#     # A_neg = add_gaussian_noise(A_star,0.5)  # guassians noise on expert actions

#     E_neg, _ = energy_head(hidden_N, A_neg,reduce="mean")  

#     with torch.no_grad():
#         margin = kappa * torch.norm((A_neg - A_star).reshape(A_neg.shape[0], -1),
#                                     dim=-1, keepdim=True)  # [B,1]
#     #     margin = F.mse_loss(A_neg, A_star)
    
    
#     # L_neg = F.mse_loss(E_neg, torch.ones_like(E_neg) * margin)

 
#     L_neg = F.relu(margin + P_loss - E_neg).mean()

#     return L_neg, E_neg.mean()


def compute_negative_energy(energy_head, A_star, layer_actions, delta, hidden_N, P_loss, pad_mask,
                            topk=2, kappa=0.6, m0=1.0):  # 新增 m0
    
    energy_head.eval()
    A_neg = layer_actions[1]
    E_neg = energy_head(hidden_N, A_neg, pad_mask, reduce="mean")

    with torch.no_grad():
        d = torch.norm((A_neg - A_star).reshape(A_neg.shape[0], -1), dim=-1, keepdim=True)  # [B,1]
        target = m0 + kappa * d + P_loss.detach()  # <-- 关键：detach 掉 E_pos
        
    L_neg = F.relu(target - E_neg).mean()
    energy_head.train()
    return L_neg, E_neg.mean()



def energy_infonce_loss(energy_model, h, a_pos, a_negs, pad_mask, tau=0.5, reduce_steps="mean"):
    """
    h:     [B,S,Dh]   
    a_pos: [B,H,Da]  
    a_negs:[B,M,H,Da]  M Negatives
    """
    B, H, Da = a_pos.shape
    M = a_negs.shape[1]

    # --- E_pos ---
    # [B,1] -> [B]
    E_pos, _ = energy_model(h, a_pos, reduce=reduce_steps)
    E_pos = E_pos.squeeze(-1)  # [B]

    # --- E_negs ---
    a_negs_flat = a_negs.reshape(B * M, H, Da)            # [B*M,H,Da]
    h_rep       = h.repeat_interleave(M, dim=0)           # [B*M,S,Dh]

    E_negs, _ = energy_model(h_rep, a_negs_flat, pad_mask, reduce=reduce_steps)  # [B*M,1]
    E_negs = E_negs.view(B, M).contiguous()               # [B,M]

    # --- EnergyNCE： -E as logits ---
    logits = torch.cat([(-E_pos).unsqueeze(1), -E_negs], dim=1) / tau  # [B,1+M]
    target = torch.zeros(B, dtype=torch.long, device=logits.device)    # expert energy index at 0

    return torch.nn.functional.cross_entropy(logits, target), E_pos.mean(), E_negs.mean()




def get_negatives(layer_actions):

    A_neg = layer_actions[1]  # (B,H,Da)
    A_neg_noise = add_gaussian_noise(A_neg,sigma=0.3) # (B,H,Da)

    return torch.cat([A_neg.unsqueeze(1),A_neg_noise.unsqueeze(1)],dim = 1)  # (B,M,H,Da)



@torch.no_grad()
def _offdiag_mask(B, device):
    return ~torch.eye(B, dtype=torch.bool, device=device)


@torch.no_grad()
def _offdiag_mask(B, device): return ~torch.eye(B, dtype=torch.bool, device=device)

def energy_inbatch_swap_infonce(
    energy_model,
    h: torch.Tensor,          # [B,S,D]
    a_pos: torch.Tensor,      # [B,H,Da]
    pad_mask: torch.Tensor,   # [B,S+H]，True=pad
    tau: float = 0.5,
    reduce_steps: str = "mean",
):
    B, S, D = h.shape
    H, Da   = a_pos.shape[1], a_pos.shape[2]
    dtype   = next(energy_model.parameters()).dtype

    h_rep = h.to(dtype).unsqueeze(1).expand(B, B, S, D).reshape(B*B, S, D)        # [B*B,S,D]
    a_rep = a_pos.to(dtype).unsqueeze(0).expand(B, B, H, Da).reshape(B*B, H, Da)  # [B*B,H,Da]
    pm = None
    if pad_mask is not None:
        pm  = pad_mask.unsqueeze(1).expand(B, B, pad_mask.size(1)).reshape(B*B, pad_mask.size(1))  # [B*B,S+H]

    E_ij = energy_model(h_rep, a_rep, pm).view(B, B, 1).squeeze(-1)               # [B,B]

    logits = (-E_ij) / tau
    labels = torch.arange(B, device=h.device)
    loss   = F.cross_entropy(logits, labels)

    E_pos_mean = torch.diag(E_ij).mean()
    E_neg_mean = E_ij[_offdiag_mask(B, h.device)].mean() if B > 1 else torch.tensor(0., device=h.device, dtype=E_ij.dtype)
    return loss, E_pos_mean, E_neg_mean




def energy_inbatch_swap_infonce_2d(
        
    energy_model,
    c_global: torch.Tensor,   # [B, D]
    a_pos: torch.Tensor,      # [B, H, Da]
    tau: float = 0.5,
    reduce_steps: str = "mean",
):
    """
    In-batch InfoNCE（2D上下文版本）。
    正样本：对每个 i，(c_i, a_i)；负样本：同一 batch 的 a_j, j≠i。
    返回: loss, E_pos_mean, E_neg_mean(仅非对角)
    """
    B, D = c_global.shape
    H, Da = a_pos.shape[1], a_pos.shape[2]
    device = c_global.device

    # 构造所有(i,j)对
    c_rep = c_global.unsqueeze(1).expand(B, B, D).reshape(B*B, D)         # [B*B, D]
    a_rep = a_pos.unsqueeze(0).expand(B, B, H, Da).reshape(B*B, H, Da)    # [B*B, H, Da]

    # 能量矩阵 E_ij
    E_ij, _ = energy_model(c_rep, a_rep, reduce=reduce_steps)  # [B*B, 1]
    E_ij = E_ij.view(B, B)                                                     # [B, B]

    # InfoNCE logits = -E / tau
    logits = (-E_ij) / tau                                                     # [B, B]
    labels = torch.arange(B, device=device)
    loss = F.cross_entropy(logits, labels)

    E_pos_mean = torch.diag(E_ij).mean()
    E_neg_mean = E_ij[_offdiag_mask(B, device)].mean() if B > 1 else torch.tensor(0., device=device)

    return loss, E_pos_mean, E_neg_mean