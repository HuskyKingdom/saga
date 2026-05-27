"""Utils for training/fine-tuning scripts."""

import torch
import torch.nn as nn

from prismatic.vla.constants import ACTION_DIM, ACTION_TOKEN_BEGIN_IDX, IGNORE_INDEX


def compute_h_loss(action_tokenizer, predicted_token_ids, ground_truth_token_ids, labels, n_embd,hnn_head,time_d):
    # pred_continuous_actions = torch.tensor(
    #     action_tokenizer.decode_token_ids_to_actions(predicted_token_ids[mask].cpu().numpy())
    # )
    # true_continuous_actions = torch.tensor(
    #     action_tokenizer.decode_token_ids_to_actions(ground_truth_token_ids[mask].cpu().numpy())
    # )
    # l1_loss = torch.nn.functional.l1_loss(pred_continuous_actions, true_continuous_actions)

    h_lambda = 0.5

    # extract action predictions from model output
    decoded_ids = action_tokenizer.decode_token_ids_to_actions(predicted_token_ids.cpu().numpy())
    action_mask = labels != -100
    filtered_ids = decoded_ids[action_mask]

    print(filtered_ids.shape,filtered_ids)
    assert 1==2
    
    z = filtered_ids
    dz_dt = None
    z_next = None
    dz_next_dt = None


    z_qp = torch.cat((z, dz_dt), dim=-1) # Shape: (B, T-2, 2*D)
    z_qp_flat = z_qp.reshape(-1, n_embd * 2) # Shape: (B*(T-2), 2*D)
    z_next_qp = torch.cat((z_next, dz_next_dt), dim=-1) # Shape: (B, T-2, 2*D)
    z_next_qp_flat = z_next_qp.reshape(-1, n_embd * 2) # Shape: (B*(T-2), 2*D)

    F1_F2_for_z = hnn_head(z_qp_flat)
    F1 = F1_F2_for_z.reshape(-1, 2)[:, 0] # Shape: (N,)
    F2 = F1_F2_for_z.reshape(-1, 2)[:, 1] # Shape: (N,)

    # hnn vector field loss
    z_qp_hat_next_flat = z_qp_flat + time_d(z_qp_flat, F1, F2) # Shape: (N, 2*D)
    z_qp_hat_next = z_qp_hat_next_flat.view(b, t-2, 2 * n_embd) # Shape: (B, T-2, 2*D)
    hnn_loss = ((z_next_qp - z_qp_hat_next)**2).mean(-1) # Shape: (B, T-1)

    hnn_reg_loss_val = h_lambda * hnn_loss.mean()
    

    return l1_loss



class Time_Derivative(nn.Module):
    """
    Calculates the HNN-inspired regularization loss.
    It takes states (for gradient calculation), their V/H potentials (pre-computed),
    and actual state changes, then computes the loss based on Hamiltonian field decomposition.
    """
    def __init__(self, n_embd):
        super().__init__()
        self.n_embd = n_embd
        self.assume_canonical_coords = True # True: assume canonical coordinates, False: use Levi-Civita tensor
        self.M = self.permutation_tensor(n_embd)
        self.field_type = 'solenoidal' # 'conservative' or 'solenoidal'

    def permutation_tensor(self,n):
        M = None
        if self.assume_canonical_coords:
            M = torch.eye(n)
            M = torch.cat([M[n//2:], -M[:n//2]])
        else:
            '''Constructs the Levi-Civita permutation tensor'''
            M = torch.ones(n,n) # matrix of ones
            M *= 1 - torch.eye(n) # clear diagonals
            M[::2] *= -1 # pattern of signs
            M[:,::2] *= -1
    
            for i in range(n): # make asymmetric
                for j in range(i+1, n):
                    M[i,j] *= -1
        return M

    def forward(self, x, F1, F2):
        # x: (N, D), requires_grad=True (comes from transformer output)
        # F1: (N,), derived from z_flat via an MLP head
        # F2: (N,), derived from z_flat via an MLP head
        F1 = F1.to(dtype=x.dtype)
        F2 = F2.to(dtype=x.dtype)

        conservative_field = 0
        solenoidal_field = 0

        # Compute the conservative fields
        if self.field_type != 'solenoidal':
            dF1 = torch.autograd.grad(F1.sum(), x, create_graph=True, allow_unused=True)[0]
            if dF1 is None:
                dF1 = torch.zeros_like(x)
            conservative_field = dF1 @ torch.eye(*self.M.shape).to(x.device).to(x.dtype)

        # Compute the solenoidal fields
        if self.field_type != 'conservative':
            dF2 = torch.autograd.grad(F2.sum(), x, create_graph=True, allow_unused=True)[0]
            if dF2 is None:
                dF2 = torch.zeros_like(x)
            solenoidal_field = dF2 @ self.M.t().to(x.device).to(x.dtype)
            
        total_field = conservative_field + solenoidal_field
        return total_field