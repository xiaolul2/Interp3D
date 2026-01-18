from typing import *
import torch
import torch.nn as nn
import torch.nn.functional as F
from .. import SparseTensor
from .full_attn import sparse_scaled_dot_product_attention
from ...attention.full_attn import scaled_dot_product_attention
from .serialized_attn import SerializeMode, sparse_serialized_scaled_dot_product_self_attention
from .windowed_attn import sparse_windowed_scaled_dot_product_self_attention
from ...attention import RotaryPositionEmbedder
import pdb
import numpy as np
from scipy.optimize import linear_sum_assignment
import math

class SparseMultiHeadRMSNorm(nn.Module):
    def __init__(self, dim: int, heads: int):
        super().__init__()
        self.scale = dim ** 0.5
        self.gamma = nn.Parameter(torch.ones(heads, dim))

    def forward(self, x: Union[SparseTensor, torch.Tensor]) -> Union[SparseTensor, torch.Tensor]:
        x_type = x.dtype
        x = x.float()
        if isinstance(x, SparseTensor):
            x = x.replace(F.normalize(x.feats, dim=-1))
        else:
            x = F.normalize(x, dim=-1)            
        return (x * self.gamma * self.scale).to(x_type)


class SparseMultiHeadAttention(nn.Module):
    def __init__(
        self,
        channels: int,
        num_heads: int,
        ctx_channels: Optional[int] = None,
        type: Literal["self", "cross"] = "self",
        attn_mode: Literal["full", "serialized", "windowed"] = "full",
        window_size: Optional[int] = None,
        shift_sequence: Optional[int] = None,
        shift_window: Optional[Tuple[int, int, int]] = None,
        serialize_mode: Optional[SerializeMode] = None,
        qkv_bias: bool = True,
        use_rope: bool = False,
        qk_rms_norm: bool = False,
    ):
        super().__init__()
        assert channels % num_heads == 0
        assert type in ["self", "cross"], f"Invalid attention type: {type}"
        assert attn_mode in ["full", "serialized", "windowed"], f"Invalid attention mode: {attn_mode}"
        assert type == "self" or attn_mode == "full", "Cross-attention only supports full attention"
        assert type == "self" or use_rope is False, "Rotary position embeddings only supported for self-attention"
        self.channels = channels
        self.ctx_channels = ctx_channels if ctx_channels is not None else channels
        self.num_heads = num_heads
        self._type = type
        self.attn_mode = attn_mode
        self.window_size = window_size
        self.shift_sequence = shift_sequence
        self.shift_window = shift_window
        self.serialize_mode = serialize_mode
        self.use_rope = use_rope
        self.qk_rms_norm = qk_rms_norm

        if self._type == "self":
            self.to_qkv = nn.Linear(channels, channels * 3, bias=qkv_bias)
        else:
            self.to_q = nn.Linear(channels, channels, bias=qkv_bias)
            self.to_kv = nn.Linear(self.ctx_channels, channels * 2, bias=qkv_bias)
        
        if self.qk_rms_norm:
            self.q_rms_norm = SparseMultiHeadRMSNorm(channels // num_heads, num_heads)
            self.k_rms_norm = SparseMultiHeadRMSNorm(channels // num_heads, num_heads)
            
        self.to_out = nn.Linear(channels, channels)

        if use_rope:
            self.rope = RotaryPositionEmbedder(channels)
        
        self.attn_phase = "normal"  # 可选值: ["normal", "store_img0", "store_img1", "interpolate"]
        self.alpha = 0.5
        self.step_id = 0
        self.layer_id = None  # 可选：如果你要区分不同层


    @staticmethod
    def _linear(module: nn.Linear, x: Union[SparseTensor, torch.Tensor]) -> Union[SparseTensor, torch.Tensor]:
        if isinstance(x, SparseTensor):
            return x.replace(module(x.feats))
        else:
            return module(x)

    @staticmethod
    def _reshape_chs(x: Union[SparseTensor, torch.Tensor], shape: Tuple[int, ...]) -> Union[SparseTensor, torch.Tensor]:
        if isinstance(x, SparseTensor):
            return x.reshape(*shape)
        else:
            return x.reshape(*x.shape[:2], *shape)

    def _fused_pre(self, x: Union[SparseTensor, torch.Tensor], num_fused: int) -> Union[SparseTensor, torch.Tensor]:
        if isinstance(x, SparseTensor):
            x_feats = x.feats.unsqueeze(0)
        else:
            x_feats = x

        #pdb.set_trace()
        x_feats = x_feats.reshape(*x_feats.shape[:2], num_fused, self.num_heads, -1)
        return x.replace(x_feats.squeeze(0)) if isinstance(x, SparseTensor) else x_feats

    def _rope(self, qkv: SparseTensor) -> SparseTensor:
        q, k, v = qkv.feats.unbind(dim=1)   # [T, H, C]
        q, k = self.rope(q, k, qkv.coords[:, 1:])
        qkv = qkv.replace(torch.stack([q, k, v], dim=1)) 
        return qkv
    
    def forward(self, x: Union[SparseTensor, torch.Tensor], context: Optional[Union[SparseTensor, torch.Tensor]] = None, store0=None, store1=None, alpha=None, step_id = 0,row_ind=None, col_ind=None, group_size=None, counter=None) -> Union[SparseTensor, torch.Tensor]:

        self.alpha=alpha
        if self._type == "self":
            qkv = self._linear(self.to_qkv, x)
            qkv = self._fused_pre(qkv, num_fused=3)
            #q, k, v = qkv.unbind(dim=1)
            if self.attn_phase == "store_img0":
                if self.use_rope:
                    qkv = self._rope(qkv)
                if self.qk_rms_norm:
                    #跑这里
                    q, k, v = qkv.unbind(dim=1)
                    with torch.no_grad():
                        store0[self.layer_id]=(k.detach(), v.detach())
                    q = self.q_rms_norm(q)
                    k = self.k_rms_norm(k)
                    qkv = qkv.replace(torch.stack([q.feats, k.feats, v.feats], dim=1))
                if self.attn_mode == "full":
                    #跑这里
                    h = sparse_scaled_dot_product_attention(qkv)
                    #pdb.set_trace()
                elif self.attn_mode == "serialized":
                    h = sparse_serialized_scaled_dot_product_self_attention(
                        qkv, self.window_size, serialize_mode=self.serialize_mode, shift_sequence=self.shift_sequence, shift_window=self.shift_window
                    )
                elif self.attn_mode == "windowed":
                    h = sparse_windowed_scaled_dot_product_self_attention(
                        qkv, self.window_size, shift_window=self.shift_window
                    )
            
            elif self.attn_phase == "store_img1":
                if self.use_rope:
                    qkv = self._rope(qkv)
                if self.qk_rms_norm:
                    #跑这里
                    q, k, v = qkv.unbind(dim=1)
                    with torch.no_grad():
                        store1[self.layer_id]=(k.detach(), v.detach())
                    q = self.q_rms_norm(q)
                    k = self.k_rms_norm(k)
                    qkv = qkv.replace(torch.stack([q.feats, k.feats, v.feats], dim=1))
                if self.attn_mode == "full":
                    #跑这里
                    h = sparse_scaled_dot_product_attention(qkv)
                elif self.attn_mode == "serialized":
                    h = sparse_serialized_scaled_dot_product_self_attention(
                        qkv, self.window_size, serialize_mode=self.serialize_mode, shift_sequence=self.shift_sequence, shift_window=self.shift_window
                    )
                elif self.attn_mode == "windowed":
                    h = sparse_windowed_scaled_dot_product_self_attention(
                        qkv, self.window_size, shift_window=self.shift_window
                    )
            elif self.attn_phase == "interpolate":
                if self.use_rope:
                    qkv = self._rope(qkv)
                if self.qk_rms_norm:
                    #跑这里
                    q, k, v = qkv.unbind(dim=1)

                    k0, v0 = store0[self.layer_id]#[selfc.step_id] # 3096 16 64
                    k1, v1 = store1[self.layer_id]#[self.step_id] #2743 16 64 #q是2727 16 64
                  
                    q = self.q_rms_norm(q)
                  
                    
                    k_out0, v_out0, k_out1, v_out1 =tokenwise_interpolation(k, v, k0, v0, k1, v1) 
                    
                    combined_k = (1-alpha)*k_out0 + alpha*k_out1 + k.feats
                    scale = (k.feats.norm() / combined_k.norm()).detach()
                    k_out = k.replace(combined_k * scale)
                    combined_v = (1-alpha)*v_out0 + alpha*v_out1 + v.feats
                    scale = (v.feats.norm() / combined_v.norm()).detach()
                    v_out = v.replace(combined_v * scale)


                    k_out = self.k_rms_norm(k_out)


                if self.attn_mode == "full":
                   
                    h = sparse_scaled_dot_product_attention(q, k_out, v_out)
                   
#
                elif self.attn_mode == "serialized":
                    h = sparse_serialized_scaled_dot_product_self_attention(
                        qkv, self.window_size, serialize_mode=self.serialize_mode, shift_sequence=self.shift_sequence, shift_window=self.shift_window
                    )
                elif self.attn_mode == "windowed":
                    h = sparse_windowed_scaled_dot_product_self_attention(
                        qkv, self.window_size, shift_window=self.shift_window
                    )
            else:
                if self.use_rope:
                    qkv = self._rope(qkv)
                if self.qk_rms_norm:
            
                    q, k, v = qkv.unbind(dim=1)
                    q = self.q_rms_norm(q)
                    k = self.k_rms_norm(k)
                    qkv = qkv.replace(torch.stack([q.feats, k.feats, v.feats], dim=1))
                if self.attn_mode == "full":
               
                    h = sparse_scaled_dot_product_attention(qkv)
                elif self.attn_mode == "serialized":
                    h = sparse_serialized_scaled_dot_product_self_attention(
                        qkv, self.window_size, serialize_mode=self.serialize_mode, shift_sequence=self.shift_sequence, shift_window=self.shift_window
                    )
                elif self.attn_mode == "windowed":
                    h = sparse_windowed_scaled_dot_product_self_attention(
                        qkv, self.window_size, shift_window=self.shift_window
                    )
        else:
            q = self._linear(self.to_q, x)
            q = self._reshape_chs(q, (self.num_heads, -1))
            kv = self._linear(self.to_kv, context)
            kv = self._fused_pre(kv, num_fused=2)
            if self.qk_rms_norm:
                q = self.q_rms_norm(q)
                k, v = kv.unbind(dim=1)
                k = self.k_rms_norm(k)
                kv = kv.replace(torch.stack([k.feats, v.feats], dim=1))
            h = sparse_scaled_dot_product_attention(q, kv)
        h = self._reshape_chs(h, (-1,))
        h = self._linear(self.to_out, h)
        return h




def tokenwise_interpolation(k, v, k0, v0, k1, v1, tau=0.6, topk=3):
    """
    不分组，直接逐 token 匹配
    - 每个 base token 与 k0/k1 中所有 token 做相似度
    - 取 top-k，超过阈值 tau 的才搬运
    """
    device = k.feats.device
    N, H, C = k.feats.shape
    D = H * C

    # flatten
    k_flat, v_flat   = k.feats.reshape(N, D),   v.feats.reshape(N, D)
    k0_flat, v0_flat = k0.feats.reshape(-1, D), v0.feats.reshape(-1, D)
    k1_flat, v1_flat = k1.feats.reshape(-1, D), v1.feats.reshape(-1, D)

    # base token normalize
    k_b_norm  = F.normalize(k_flat, dim=-1)
    k0_norm   = F.normalize(k0_flat, dim=-1)
    k1_norm   = F.normalize(k1_flat, dim=-1)

    # sim (N_base × N_side)
    sim0 = k_b_norm @ k0_norm.T
    sim1 = k_b_norm @ k1_norm.T

    def match_and_copy(sim, k_side, v_side):
        topv, topi = torch.topk(sim, min(topk, sim.size(1)), dim=-1)
        # 默认只取第一个符合条件的
        chosen_idx = torch.full((sim.size(0),), -1, device=device, dtype=torch.long)
        mask = topv[:, 0] > tau
        chosen_idx[mask] = topi[mask, 0]

        k_out = k_flat.clone()
        v_out = v_flat.clone()
        valid = chosen_idx >= 0
        k_out[valid] = k_side[chosen_idx[valid]]
        v_out[valid] = v_side[chosen_idx[valid]]
        return k_out, v_out

    k_out0, v_out0 = match_and_copy(sim0, k0_flat, v0_flat)
    k_out1, v_out1 = match_and_copy(sim1, k1_flat, v1_flat)

    return (
        k_out0.view(N, H, C),
        v_out0.view(N, H, C),
        k_out1.view(N, H, C),
        v_out1.view(N, H, C),
    )