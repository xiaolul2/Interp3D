from typing import *
import torch
import torch.nn as nn
import torch.nn.functional as F
from .full_attn import scaled_dot_product_attention
import pdb
from scipy.optimize import linear_sum_assignment

 
class MultiHeadRMSNorm(nn.Module):
    def __init__(self, dim: int, heads: int):
        super().__init__()
        self.scale = dim ** 0.5
        self.gamma = nn.Parameter(torch.ones(heads, dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (F.normalize(x.float(), dim = -1) * self.gamma * self.scale).to(x.dtype)


class RotaryPositionEmbedder(nn.Module):
    def __init__(self, hidden_size: int, in_channels: int = 3):
        super().__init__()
        assert hidden_size % 2 == 0, "Hidden size must be divisible by 2"
        self.hidden_size = hidden_size
        self.in_channels = in_channels
        self.freq_dim = hidden_size // in_channels // 2
        self.freqs = torch.arange(self.freq_dim, dtype=torch.float32) / self.freq_dim
        self.freqs = 1.0 / (10000 ** self.freqs)
        
    def _get_phases(self, indices: torch.Tensor) -> torch.Tensor:
        self.freqs = self.freqs.to(indices.device)
        phases = torch.outer(indices, self.freqs)
        phases = torch.polar(torch.ones_like(phases), phases)
        return phases
        
    def _rotary_embedding(self, x: torch.Tensor, phases: torch.Tensor) -> torch.Tensor:
        x_complex = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))
        x_rotated = x_complex * phases
        x_embed = torch.view_as_real(x_rotated).reshape(*x_rotated.shape[:-1], -1).to(x.dtype)
        return x_embed
        
    def forward(self, q: torch.Tensor, k: torch.Tensor, indices: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            q (sp.SparseTensor): [..., N, D] tensor of queries
            k (sp.SparseTensor): [..., N, D] tensor of keys
            indices (torch.Tensor): [..., N, C] tensor of spatial positions
        """
        if indices is None:
            indices = torch.arange(q.shape[-2], device=q.device)
            if len(q.shape) > 2:
                indices = indices.unsqueeze(0).expand(q.shape[:-2] + (-1,))
        
        phases = self._get_phases(indices.reshape(-1)).reshape(*indices.shape[:-1], -1)
        if phases.shape[1] < self.hidden_size // 2:
            phases = torch.cat([phases, torch.polar(
                torch.ones(*phases.shape[:-1], self.hidden_size // 2 - phases.shape[1], device=phases.device),
                torch.zeros(*phases.shape[:-1], self.hidden_size // 2 - phases.shape[1], device=phases.device)
            )], dim=-1)
        q_embed = self._rotary_embedding(q, phases)
        k_embed = self._rotary_embedding(k, phases)
        return q_embed, k_embed
    


class AttentionProcessor:
    def __call__(self, attn_module, x, context=None, indices=None):
        raise NotImplementedError


class DefaultAttentionProcessor(AttentionProcessor):
    def __call__(self, attn_module, x, context=None, indices=None):
        return attn_module._forward(x, context, indices)


class MultiHeadAttention(nn.Module):
    def __init__(  
        self,
        channels: int,
        num_heads: int,
        ctx_channels: Optional[int]=None,
        type: Literal["self", "cross"] = "self",
        attn_mode: Literal["full", "windowed"] = "full",
        window_size: Optional[int] = None,
        shift_window: Optional[Tuple[int, int, int]] = None,
        qkv_bias: bool = True,
        use_rope: bool = False,
        qk_rms_norm: bool = False,
    ):
        super().__init__()
        assert channels % num_heads == 0
        assert type in ["self", "cross"], f"Invalid attention type: {type}"
        assert attn_mode in ["full", "windowed"], f"Invalid attention mode: {attn_mode}"
        assert type == "self" or attn_mode == "full", "Cross-attention only supports full attention"
        
        if attn_mode == "windowed":
            raise NotImplementedError("Windowed attention is not yet implemented")
        
        self.channels = channels
        self.head_dim = channels // num_heads
        self.ctx_channels = ctx_channels if ctx_channels is not None else channels
        self.num_heads = num_heads
        self._type = type
        self.attn_mode = attn_mode
        self.window_size = window_size
        self.shift_window = shift_window
        self.use_rope = use_rope
        self.qk_rms_norm = qk_rms_norm

        if self._type == "self":
            self.to_qkv = nn.Linear(channels, channels * 3, bias=qkv_bias)
        else:
            self.to_q = nn.Linear(channels, channels, bias=qkv_bias)
            self.to_kv = nn.Linear(self.ctx_channels, channels * 2, bias=qkv_bias)
            
        if self.qk_rms_norm:
            self.q_rms_norm = MultiHeadRMSNorm(self.head_dim, num_heads)
            self.k_rms_norm = MultiHeadRMSNorm(self.head_dim, num_heads)
            
        self.to_out = nn.Linear(channels, channels)

        if use_rope:
            self.rope = RotaryPositionEmbedder(channels)
        
        
        self.attn_phase = "normal"  # 可选值: ["normal", "store_img0", "store_img1", "interpolate"]
        self.alpha = 0.5
        self.step_id = 0
        self.layer_id = None  # 可选：如果你要区分不同层
    
    def forward(self, x: torch.Tensor, context: Optional[torch.Tensor] = None, indices: Optional[torch.Tensor] = None, store0=None, store1=None, alpha=None, step_id = 0,row_ind=None, col_ind=None, group_size=None) -> torch.Tensor:
         
        B, L, C = x.shape
        #pdb.set_trace()
        if alpha!=None:
            self.alpha = alpha
        if self._type == "self":
            #self.step_id+=1q
            # 
            qkv = self.to_qkv(x)
            qkv = qkv.reshape(B, L, 3, self.num_heads, -1)
            q, k, v = qkv.unbind(dim=2)
            # 
            # ========= 插值/存储机制 ========= #
            if self.attn_phase == "store_img0":
                #print(self.attn_phase)
                #print(self.step_id)
                #if self.step_id<=23:
                # 
                with torch.no_grad():
                    store0[self.layer_id]=(k.detach(), v.detach())
                if self.use_rope: 
                    q, k = self.rope(q, k, indices)
                
                #qkv = torch.stack([q, k, v], dim=2)
                if self.attn_mode == "full":
                     
                    if self.qk_rms_norm:
                        #q, k, v = qkv.unbind(dim=2)
                        q = self.q_rms_norm(q)
                        k = self.k_rms_norm(k)
                        h = scaled_dot_product_attention(q, k, v)
                    else:
                        h = scaled_dot_product_attention(qkv)

            elif self.attn_phase == "store_img1":
                #print(self.attn_phase)
                # 
                with torch.no_grad():
                    store1[self.layer_id]=(k.detach(), v.detach())
#
                if self.use_rope: 
                    q, k = self.rope(q, k, indices)
                    
                
                #qkv = torch.stack([q, k, v], dim=2)
                if self.attn_mode == "full":
                    if self.qk_rms_norm:
                        #pdb.set_trace()
                        #q, k, v = qkv.unbind(dim=2)
                        q = self.q_rms_norm(q)
                        k = self.k_rms_norm(k)
                        h = scaled_dot_product_attention(q, k, v)
                    else:
                        h = scaled_dot_product_attention(qkv)
#
            elif self.attn_phase == "interpolate":
               
                #print(self.attn_phase)
                k0, v0 = store0[self.layer_id]#[selfc.step_id]
                k1, v1 = store1[self.layer_id]#[self.step_id]
                
                #replace
                #k = (1 - self.alpha) * k0 + self.alpha * k1
                #v = (1 - self.alpha) * v0 + self.alpha * v1
#
                #inner
                ##pdb.set_trace()
                #k = torch.cat([(1 - self.alpha) * k0 + self.alpha * k1,k],dim=1)
                #v = torch.cat([(1 - self.alpha) * v0 + self.alpha * v1,v], dim=1)
#
                #outer

                k_0 = torch.cat([k0,k], dim=1)
                #pdb.set_trace()
                v_0 = torch.cat([v0,v], dim=1)
###############
                k_1 = torch.cat([k1,k], dim=1)
                v_1 = torch.cat([v1,v], dim=1)

            # =================================== #
                if self.use_rope: 
                    q, k = self.rope(q, k, indices) 
#
                #qkv = torch.stack([q, k, v], dim=2)
                if self.attn_mode == "full":
                    if self.qk_rms_norm:
                        #q, k, v = qkv.unbind(dim=2)
                        #q0 = self.q_rms_norm(q0)
                        q = self.q_rms_norm(q)
##
                        #k = self.k_rms_norm(k)
                        #h = scaled_dot_product_attention(q, k, v)
#
# 
                        #outer
                        k_1 = self.k_rms_norm(k_1)
                        k_0 = self.k_rms_norm(k_0)
                        #####k = self.k_rms_norm(k)
                        h_0 = scaled_dot_product_attention(q, k_0, v_0)
                        h_1 = scaled_dot_product_attention(q, k_1, v_1)
                        #print(self.alpha)
                        #h_0, h_1 = reorder_q_sequence(h_0, h_1, self.alpha)
                        h = (1 - self.alpha) * h_0 + self.alpha * h_1
#
                    else:
                        h = scaled_dot_product_attention(qkv)
            else:
                if self.use_rope: 
                    q, k, v = qkv.unbind(dim=2)
                    q, k = self.rope(q, k, indices)
                    qkv = torch.stack([q, k, v], dim=2)

                #qkv = torch.stack([q, k, v], dim=2)
                if self.attn_mode == "full":
                    if self.qk_rms_norm:
                        q, k, v = qkv.unbind(dim=2)
                        q = self.q_rms_norm(q)
                        k = self.k_rms_norm(k)
                        h = scaled_dot_product_attention(q, k, v)
                    else:
                        h = scaled_dot_product_attention(qkv)
            #self.step_id+=1
        else:
            Lkv = context.shape[1]
            q = self.to_q(x)
            kv = self.to_kv(context)
            q = q.reshape(B, L, self.num_heads, -1)
            kv = kv.reshape(B, Lkv, 2, self.num_heads, -1)
            k, v = kv.unbind(dim=2)
            if self.qk_rms_norm:
                q = self.q_rms_norm(q)
                k = self.k_rms_norm(k)
                h = scaled_dot_product_attention(q, k, v)
            else:
                h = scaled_dot_product_attention(q, kv)


            


        h = h.reshape(B, L, -1)
        h = self.to_out(h)
        #pdb.set_trace()
        return h


from torch import FloatTensor, LongTensor, Size, Tensor
def slerp(A, B, a):
    """
    Spherical Linear Interpolation between A and B
    a: float or tensor, interpolation coefficient between 0 and 1
    A, B: torch tensors of shape [..., D] (can be batched)
    Return: interpolated tensor of same shape
    """
    # Normalize to unit vectors
    A_norm = F.normalize(A, dim=-1)
    B_norm = F.normalize(B, dim=-1)

    # Dot product and angle
    dot = (A_norm * B_norm).sum(dim=-1, keepdim=True).clamp(-1.0, 1.0)
    theta = torch.acos(dot)

    sin_theta = torch.sin(theta)
    sin_theta = torch.where(sin_theta == 0, torch.ones_like(sin_theta), sin_theta)  # avoid /0

    s1 = torch.sin((1 - a) * theta) / sin_theta
    s2 = torch.sin(a * theta) / sin_theta

    return s1 * A + s2 * B
