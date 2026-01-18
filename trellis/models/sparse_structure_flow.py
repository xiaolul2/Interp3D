from typing import *
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from ..modules.utils import convert_module_to_f16, convert_module_to_f32
from ..modules.transformer import AbsolutePositionEmbedder, ModulatedTransformerCrossBlock
from ..modules.spatial import patchify, unpatchify

import pdb
class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        """
        Create sinusoidal timestep embeddings.

        Args:
            t: a 1-D Tensor of N indices, one per batch element.
                These may be fractional.
            dim: the dimension of the output.
            max_period: controls the minimum frequency of the embeddings.

        Returns:
            an (N, D) Tensor of positional embeddings.
        """
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        half = dim // 2
        freqs = torch.exp(
            -np.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb


class SparseStructureFlowModel(nn.Module):
    def __init__(
        self,
        resolution: int,
        in_channels: int,
        model_channels: int,
        cond_channels: int,
        out_channels: int,
        num_blocks: int,
        num_heads: Optional[int] = None,
        num_head_channels: Optional[int] = 64,
        mlp_ratio: float = 4,
        patch_size: int = 2,
        pe_mode: Literal["ape", "rope"] = "ape",
        use_fp16: bool = False,
        use_checkpoint: bool = False,
        share_mod: bool = False,
        qk_rms_norm: bool = False,
        qk_rms_norm_cross: bool = False,
    ):
        super().__init__()
        self.resolution = resolution
        self.in_channels = in_channels
        self.model_channels = model_channels
        self.cond_channels = cond_channels
        self.out_channels = out_channels
        self.num_blocks = num_blocks
        self.num_heads = num_heads or model_channels // num_head_channels
        self.mlp_ratio = mlp_ratio
        self.patch_size = patch_size
        self.pe_mode = pe_mode
        self.use_fp16 = use_fp16
        self.use_checkpoint = use_checkpoint
        self.share_mod = share_mod
        self.qk_rms_norm = qk_rms_norm
        self.qk_rms_norm_cross = qk_rms_norm_cross
        self.dtype = torch.float16 if use_fp16 else torch.float32

        self.t_embedder = TimestepEmbedder(model_channels)
        self.kv_dict_img0 = DefaultDict(list)
        self.kv_dict_img1 = DefaultDict(list)
        #self.h_store0 = DefaultDict(list)
        #self.h_store1 = DefaultDict(list)
        #self.kv_dict_img0c = DefaultDict(list)
        #self.kv_dict_img1c = DefaultDict(list)
        
        #self.token_dict0 = DefaultDict(list)
        #self.token_dict1 = DefaultDict(list)

        if share_mod:
            self.adaLN_modulation = nn.Sequential(
                nn.SiLU(),
                nn.Linear(model_channels, 6 * model_channels, bias=True)
            )

        if pe_mode == "ape":
            pos_embedder = AbsolutePositionEmbedder(model_channels, 3)
            coords = torch.meshgrid(*[torch.arange(res, device=self.device) for res in [resolution // patch_size] * 3], indexing='ij')
            coords = torch.stack(coords, dim=-1).reshape(-1, 3)
            pos_emb = pos_embedder(coords)
            self.register_buffer("pos_emb", pos_emb)

        self.input_layer = nn.Linear(in_channels * patch_size**3, model_channels) 
            
        self.blocks = nn.ModuleList([
            ModulatedTransformerCrossBlock(
                model_channels,
                cond_channels,
                num_heads=self.num_heads,
                mlp_ratio=self.mlp_ratio,
                attn_mode='full',
                use_checkpoint=self.use_checkpoint,
                use_rope=(pe_mode == "rope"),
                share_mod=share_mod,
                qk_rms_norm=self.qk_rms_norm,
                qk_rms_norm_cross=self.qk_rms_norm_cross,
            )
            for _ in range(num_blocks)
        ])

        self.out_layer = nn.Linear(model_channels, out_channels * patch_size**3)

        self.initialize_weights()
        if use_fp16:
            self.convert_to_fp16()
        self.mode = None
        self.step_id0 = 0
        self.step_id1 = 0
        self.step_id2 = 0

        self.alpha=None
    @property
    def device(self) -> torch.device:
        """
        Return the device of the model.
        """
        return next(self.parameters()).device

    def convert_to_fp16(self) -> None:
        """
        Convert the torso of the model to float16.
        """
        self.blocks.apply(convert_module_to_f16)

    def convert_to_fp32(self) -> None:
        """
        Convert the torso of the model to float32.
        """
        self.blocks.apply(convert_module_to_f32)

    def initialize_weights(self) -> None:
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)

        # Initialize timestep embedding MLP:
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

        # Zero-out adaLN modulation layers in DiT blocks:
        if self.share_mod:
            nn.init.constant_(self.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(self.adaLN_modulation[-1].bias, 0)
        else:
            for block in self.blocks:
                nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
                nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        # Zero-out output layers:
        nn.init.constant_(self.out_layer.weight, 0)
        nn.init.constant_(self.out_layer.bias, 0)

    def forward(self, x, t, cond, cond_01, alpha=None,row_ind=None, col_ind=None, group_size=None, counter=None) -> torch.Tensor:
        assert [*x.shape] == [x.shape[0], self.in_channels, *[self.resolution] * 3], \
                f"Input shape mismatch, got {x.shape}, expected {[x.shape[0], self.in_channels, *[self.resolution] * 3]}"
        #pdb.set_trace()
        #if self.mode == "store_img0":
            #pdb.set_qtrace()
        #pdb.set_trace()
        h = patchify(x, self.patch_size)
        h = h.view(*h.shape[:2], -1).permute(0, 2, 1).contiguous()
        h = self.input_layer(h)
        h = h + self.pos_emb[None]
        t_emb = self.t_embedder(t)
        if self.share_mod:
            t_emb = self.adaLN_modulation(t_emb)
        t_emb = t_emb.type(self.dtype)
        h = h.type(self.dtype)
        cond = cond.type(self.dtype)
        #pdb.set_trace()
        for i in range(len(self.blocks)):
            #pdb.set_trace()
            h = self.blocks[i](h, t_emb, cond, self.kv_dict_img0, self.kv_dict_img1,  alpha, self.step_id0, row_ind, col_ind,group_size)
        h = h.type(x.dtype)
        h = F.layer_norm(h, h.shape[-1:])
        h = self.out_layer(h)
        ###pdb.set_trace()
        h = h.permute(0, 2, 1).view(h.shape[0], h.shape[2], *[self.resolution // self.patch_size] * 3)
        h = unpatchify(h, self.patch_size).contiguous()
        #self.h_store0[f"step_{self.step_id0}"] = h.detach()
        #self.step_id0+=1
        #self.h_store0['h'] = h.detach()
        
        return h

def reorder_token_sequences(cond_src: torch.Tensor, cond_tgt: torch.Tensor, alpha: float = 0.5):
    """
    Reorders token sequences by aligning tokens with nearest cosine similarity pairs.
    
    Args:
        cond_src: [1, N, D] - source tokens
        cond_tgt: [1, N, D] - target tokens
        alpha: float in [0,1], controls reordering direction:
               if alpha < 0.5: reorder target to match source;
               else: reorder source to match target.
    
    Returns:
        reordered_src: [1, N, D]
        reordered_tgt: [1, N, D]
    """
    assert cond_src.shape == cond_tgt.shape, "Source and target token shapes must match"
    assert cond_src.shape[0] == 1, "Only batch size 1 is supported"
    
    cond_src = cond_src[0]  # [N, D]
    cond_tgt = cond_tgt[0]  # [N, D]
    
    # Normalize features to unit vectors
    src_norm = F.normalize(cond_src, dim=-1)  # [N, D]
    tgt_norm = F.normalize(cond_tgt, dim=-1)  # [N, D]
    
    # Compute cosine distance matrix: 1 - cosine similarity
    dists = 1 - torch.matmul(src_norm, tgt_norm.T)  # [N, N]

    indices0 = dists.argmin(dim=1)  # [N]
    reordered_src = cond_src[indices0]
    indices1 = dists.argmin(dim=0) 
    reordered_tgt = cond_tgt[indices1]
    
    #if alpha < 0.5:
    #    # Reorder target to match source
    #    indices = dists.argmin(dim=1)  # [N]
    #    reordered_src = cond_src
    #    reordered_tgt = cond_tgt[indices]
    #elif alpha == 0.5:
    #    indices0 = dists.argmin(dim=1)  # [N]
    #    reordered_src = cond_src[indices0]
    #    indices1 = dists.argmin(dim=0) 
    #    reordered_tgt = cond_tgt[indices1]
    #else:
    #    # Reorder source to match target
    #    indices = dists.argmin(dim=0)  # [N]
    #    reordered_src = cond_src[indices]
    #    reordered_tgt = cond_tgt

    return reordered_src.unsqueeze(0), reordered_tgt.unsqueeze(0)  # [1, N, D]