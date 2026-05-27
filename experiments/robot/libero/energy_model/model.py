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
        hN: [B, S, D_h] or [S, D_h], a: [B, H,  D_a]
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
        
        # Ensure hN has batch dimension if it's 2-D
        if hN.dim() == 2:
            hN = hN.unsqueeze(0)  # [S, D_h] -> [1, S, D_h]
        
        # Ensure a has batch dimension if it's 2-D
        if a.dim() == 2:
            a = a.unsqueeze(0)  # [H, D_a] -> [1, H, D_a]
        
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

        # Adjust pad_mask to match context_mapped sequence length
        if pad_mask is not None:
            B, S_context = context_mapped.shape[:2]
            
            # Ensure pad_mask is 2-D [B, S]
            if pad_mask.dim() == 1:
                pad_mask = pad_mask.unsqueeze(0).expand(B, -1)
            
            pad_mask_seq_len = pad_mask.shape[1]
            
            if pad_mask_seq_len != S_context:
                # If pad_mask is longer, take the first S_context elements
                # This assumes hN is a subset of the full sequence (e.g., only vision patches)
                if pad_mask_seq_len > S_context:
                    pad_mask = pad_mask[:, :S_context]
                else:
                    # If pad_mask is shorter, pad with False (no masking for extra positions)
                    pad_mask_padded = torch.zeros(B, S_context, dtype=pad_mask.dtype, device=pad_mask.device)
                    pad_mask_padded[:, :pad_mask_seq_len] = pad_mask
                    pad_mask = pad_mask_padded

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
