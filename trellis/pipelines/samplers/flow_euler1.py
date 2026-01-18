from typing import *

from huggingface_hub.inference._generated.types import feature_extraction
import torch
import numpy as np
from tqdm import tqdm
from easydict import EasyDict as edict
from .base import Sampler
from .classifier_free_guidance_mixin import ClassifierFreeGuidanceSamplerMixin
from .guidance_interval_mixin import GuidanceIntervalSamplerMixin


from scipy.spatial import cKDTree
from sklearn.cluster import KMeans
from scipy.optimize import linear_sum_assignment
import torch.nn.functional as F
import copy
import pdb
from torch import FloatTensor, LongTensor, Size, Tensor
from sklearn.cluster import MiniBatchKMeans
import time
from sklearn.decomposition import PCA


import random
import numpy as np

random.seed(1)
np.random.seed(1)
torch.manual_seed(1)
torch.cuda.manual_seed_all(1)

class FlowEulerSampler(Sampler):
    """
    Generate samples from a flow-matching model using Euler sampling.

    Args:
        sigma_min: The minimum scale of noise in flow.
    """
    def __init__(
        self,
        sigma_min: float,
    ):
        self.sigma_min = sigma_min

    def _eps_to_xstart(self, x_t, t, eps):
        assert x_t.shape == eps.shape
        return (x_t - (self.sigma_min + (1 - self.sigma_min) * t) * eps) / (1 - t)

    def _xstart_to_eps(self, x_t, t, x_0):
        assert x_t.shape == x_0.shape
        return (x_t - (1 - t) * x_0) / (self.sigma_min + (1 - self.sigma_min) * t)

    def _v_to_xstart_eps(self, x_t, t, v):
        assert x_t.shape == v.shape
        eps = (1 - t) * v + x_t
        x_0 = (1 - self.sigma_min) * x_t - (self.sigma_min + (1 - self.sigma_min) * t) * v
        return x_0, eps

    def _inference_model(self, model, x_t, t, cond=None, cond_01=None, alpha=None,row_ind=None, col_ind=None, group_size=None, counter=None, **kwargs):
        
        t = torch.tensor([1000 * t] * x_t.shape[0], device=x_t.device, dtype=torch.float32)
        if cond is not None and cond.shape[0] == 1 and x_t.shape[0] > 1:
            cond = cond.repeat(x_t.shape[0], *([1] * (len(cond.shape) - 1)))

        return model(x_t, t, cond, cond_01, alpha, row_ind, col_ind, group_size, counter)

    def _get_model_prediction(self, model, x_t, t, cond=None, cond_01=None, alpha=None, row_ind=None, col_ind=None, group_size=None,counter=None,**kwargs):
        
        pred_v = self._inference_model(model=model, x_t=x_t, t=t, cond=cond, cond_01=cond_01, alpha=alpha, row_ind=row_ind, col_ind=col_ind, group_size=group_size,counter=counter, **kwargs)

        pred_x_0, pred_eps = self._v_to_xstart_eps(x_t=x_t, t=t, v=pred_v)
        return pred_x_0, pred_eps, pred_v

    @torch.no_grad()
    def sample_once(
        self,
        model,
        x_t,
        t: float,
        t_prev: float,
        cond: Optional[Any] = None,
        cond_01: Optional[Any] = None,
        alpha: Optional[Any] = None,
        row_ind:Optional[Any] = None,
        col_ind:Optional[Any] = None,
        group_size:Optional[Any]=None,
        counter:Optional[Any]=None,
        **kwargs
    ):
        """
        Sample x_{t-1} from the model using Euler method.
        
        Args:
            model: The model to sample from.
            x_t: The [N x C x ...] tensor of noisy inputs at time t.
            t: The current timestep.
            t_prev: The previous timestep.
            cond: conditional information.
            **kwargs: Additional arguments for model inference.

        Returns:
            a dict containing the following
            - 'pred_x_prev': x_{t-1}.
            - 'pred_x_0': a prediction of x_0.
        """
        #if cond_01 == None:
        ###
        pred_x_0, pred_eps, pred_v = self._get_model_prediction(model, x_t, t, cond, cond_01, alpha, row_ind, col_ind, group_size,counter, **kwargs)
        pred_x_prev = x_t - (t - t_prev) * pred_v
        return edict({"pred_x_prev": pred_x_prev, "pred_x_0": pred_x_0})

    @torch.no_grad()
    def sample(
        self,
        is_structure,
        z0,z1,alpha_list,
        model,
        noise,
        cond_01: Optional[Any] = None,
        cond: Optional[Any] = None,
        steps: int = 50,
        rescale_t: float = 1.0,
        verbose: bool = True,
        **kwargs
    ):
        """
        Generate samples from the model using Euler method.
        
        Args:
            model: The model to sample from.
            noise: The initial noise tensor.
            cond: conditional information.
            steps: The number of steps to sample.
            rescale_t: The rescale factor for t.
            verbose: If True, show a progress bar.
            **kwargs: Additional arguments for model_inference.

        Returns:
            a dict containing the following
            - 'samples': the model samples.
            - 'pred_x_t': a list of prediction of x_t.
            - 'pred_x_0': a list of prediction of x_0.
        """
   
        #if cond_01!=None:
        def set_attn_mode(model, mode: str, alpha: float = 0.5):
            model.mode = mode
            model.step_id0 = 0
            model.step_id1 = 0
            model.step_id2 = 0
            model.alpha = alpha
            for i, block in enumerate(model.blocks):
                attn = block.self_attn
                attn.attn_phase = mode
                attn.layer_id = f"block_{i}"
                #if mode == "interpolate":
                #    attn.alpha = alpha
                if mode == "store_img0":
                    model.kv_dict_img0.clear()
                if mode == "store_img1":
                    model.kv_dict_img1.clear()

        sample = noise
        if alpha_list!=None:
            alpha_list = alpha_list[1:-1]
        else:
            alpha_list = [None, None,  None]
        
        #alpha_list = [0.16667, 0.33333, 0.5, 0.666667, 0.833333]

       
        t_seq = np.linspace(1, 0, steps + 1)
        t_seq = rescale_t * t_seq / (1 + (rescale_t - 1) * t_seq)
        t_pairs = list((t_seq[i], t_seq[i + 1]) for i in range(steps))
        ret = edict({"samples": None, "pred_x_t": [], "pred_x_0": []})
        ret_list = [edict({"samples": None, "pred_x_t": [], "pred_x_0": []}) for i in range(len(alpha_list))]

        #这边可以用不同的更改方法
        if cond_01!=None:
            if is_structure:
                sample_start = noise
                sample_end = noise
                sample_list= [noise for i in range(len(alpha_list))]
            else:
                sample_start = noise[0] 
                sample_end = noise[-1]
                sample_list = noise[1:-1]
        
            #inner_alpha_list0 = list(torch.linspace(cond_01['alpha'], 0.5, len(t_pairs)).to(sample_list[0].device))
            #inner_alpha_list1 = list(torch.linspace(cond_01['alpha'], 0.5, len(t_pairs)).to(sample_list[0].device))
            #inner_group_list = list(torch.linspace( 32,1, len(t_pairs)).to(sample_list[0].device))
            steps = torch.linspace(0, 1, len(t_pairs)).to(sample_list[0].device)
            inner_group_list = list(32 * (0.001 ** (1-steps).flip(0)) ) 
            inner_group_list1 = list(64 * (0.001 ** (1-steps).flip(0)) ) 
           
            #inner_alpha_list_all =[list(torch.linspace(alpha_list[i], 0.5, len(t_pairs)).to(sample_list[0].device)) for i in range(len(alpha_list))]
            #n = len(t_pairs)
            ### 指数型权重 (前慢后快)
            #weights = torch.linspace(0, 1, n).pow(2).to(sample_list[0].device)   # [0,1] 映射到 [0,1]，但前半区更密集
            #inner_alpha_list_all = [
            #    list((alpha_list[i] + (0.5 - alpha_list[i]) * weights))
            #    for i in range(len(alpha_list))
            #]

            n = len(t_pairs)
            # 指数型权重 (前慢后快)
            gamma = 2.0  # 控制加速强度，越大越后快
            t = torch.linspace(0, 1, n, device=alpha_list[0].device)
#
            # 指数型权重，范围 [0,1]
            w = (torch.exp(gamma * t) - 1) / (torch.exp(torch.tensor(gamma, device=t.device)) - 1)
#
            inner_alpha_list_all = [
                list(alpha_list[i] + (0.5 - alpha_list[i]) * w)
                for i in range(len(alpha_list))
            ]

          
            cond_start = cond_01['cond_start']
            cond_end = cond_01['cond_end']
            cond_start_new = copy.deepcopy(cond_start)
            cond_end_new = copy.deepcopy(cond_end)

            #cond_start_new = cond_start
            #cond_end_new = cond_end
            ###0801
            

            


        counter=0
        #sample0 = 
        out_list = [None for i in range(len(alpha_list))]
        
        for t, t_prev in tqdm(t_pairs, desc="Sampling", disable=not verbose):
            if cond_01!=None:
                if is_structure:
                    set_attn_mode(model, mode="store_img0")
         
                    out_0 = self.sample_once(model, sample_start, t, t_prev, cond_start["cond"], None, None,None,None, None,**kwargs)
    
                    sample_start = out_0.pred_x_prev

                    set_attn_mode(model, mode="store_img1")
                    out_1 = self.sample_once(model, sample_end, t, t_prev, cond_end["cond"], None, None,None,None, None,**kwargs)

                    

                    sample_end = out_1.pred_x_prev
              
                    group_size1 = int(torch.ceil(inner_group_list1[counter]))
                    group_size = int(torch.ceil(inner_group_list[counter]))

                  
                    idx0 = downsample_then_match_multi_group_new(z0, z1, grid_size=64, target_size=16, group_size=group_size1)
                    reorder_kv_by_idx_sparse1_new(idx0, model)
                    
                    row_ind, col_ind = None, None
                
                else:
                    row_ind, col_ind = None, None
                    #group_size = int(torch.ceil(inner_group_list[counter]))
                    set_attn_mode(model, mode="store_img0")
                    out_0 = self.sample_once(model, sample_start, t, t_prev, cond_start["cond"], None, None,None,None,None, **kwargs)
                    sample_start = out_0.pred_x_prev
                    set_attn_mode(model, mode="store_img1")
                    out_1 = self.sample_once(model, sample_end, t, t_prev, cond_end["cond"], None, None,None,None, None,**kwargs)
                    sample_end = out_1.pred_x_prev
                    #获得sample_start和sample_end的
                    #pdb.set_trace()
                    group_size = int(torch.ceil(inner_group_list[counter]))
                
                
                for i in range(len(alpha_list)):
                    if alpha_list[i]==0.5:
                        alpha = 0.5#inner_alpha_list_all[i][counter]
                    else:
                        alpha = inner_alpha_list_all[i][counter]

                    #alpha_list =  np.linspace(0,1,7)[1:-1]
                    alpha = alpha_list[i]

                    cond_start_new['cond'], cond_end_new['cond'] = group_and_reorder_by_group_similarity( cond_start['cond'], cond_end['cond'], group_size=group_size)
                    cond =(1-alpha) * cond_start_new['cond'] + alpha * cond_end_new['cond']
                    
            
                    if is_structure:
                        set_attn_mode(model, mode="interpolate", alpha=alpha)
                     
                        out_list[i] = self.sample_once(model, sample_list[i], t, t_prev, cond, cond_01, alpha, row_ind, col_ind, group_size, counter, **kwargs)
                        sample_list[i] = out_list[i].pred_x_prev
                    else:
                        set_attn_mode(model, mode="interpolate", alpha=alpha)
                        #alpha = alpha_list[i]
                        out_list[i] = self.sample_once(model, sample_list[i], t, t_prev, cond, cond_01, alpha, None, None, None,counter, **kwargs)
                        sample_list[i] = out_list[i].pred_x_prev
                        
                    
                    ret_list[i].pred_x_t.append(out_list[i].pred_x_prev)
                    ret_list[i].pred_x_0.append(out_list[i].pred_x_0)
                    
            else:
                out = self.sample_once(model, sample, t, t_prev, cond, None, None, None, None, None,None,**kwargs)
                sample = out.pred_x_prev
                ret.pred_x_t.append(out.pred_x_prev)
                ret.pred_x_0.append(out.pred_x_0)
            counter+=1
        if out_list[0]==None:  
            ret.samples = sample
            return ret
        else:
            for i in range(len(alpha_list)):
                ret_list[i].samples = sample_list[i]
            return ret_list


class FlowEulerCfgSampler(ClassifierFreeGuidanceSamplerMixin, FlowEulerSampler):
    """
    Generate samples from a flow-matching model using Euler sampling with classifier-free guidance.
    """
    @torch.no_grad()
    def sample(
        self,
        model,
        noise,
        cond,
        neg_cond,
        steps: int = 50,
        rescale_t: float = 1.0,
        cfg_strength: float = 3.0,
        verbose: bool = True,
        **kwargs
    ):
        """
        Generate samples from the model using Euler method.
        
        Args:
            model: The model to sample from.
            noise: The initial noise tensor.
            cond: conditional information.
            neg_cond: negative conditional information.
            steps: The number of steps to sample.
            rescale_t: The rescale factor for t.
            cfg_strength: The strength of classifier-free guidance.
            verbose: If True, show a progress bar.
            **kwargs: Additional arguments for model_inference.

        Returns:
            a dict containing the following
            - 'samples': the model samples.
            - 'pred_x_t': a list of prediction of x_t.
            - 'pred_x_0': a list of prediction of x_0.
        """
        return super().sample(model, noise, cond, steps, rescale_t, verbose, neg_cond=neg_cond, cfg_strength=cfg_strength, **kwargs)


class FlowEulerGuidanceIntervalSampler(GuidanceIntervalSamplerMixin, FlowEulerSampler):
    """
    Generate samples from a flow-matching model using Euler sampling with classifier-free guidance and interval.
    """
    @torch.no_grad()
    def sample(
        self,
        is_structure,
        z0,z1,alpha_list,
        model,
        noise,
        cond_01,
        cond,
        neg_cond,
        steps: int = 50,
        rescale_t: float = 1.0,
        cfg_strength: float = 3.0,
        cfg_interval: Tuple[float, float] = (0.0, 1.0),
        verbose: bool = True,
        **kwargs
    ):
     ##3D structure 跑的这个
        """
        Generate samples from the model using Euler method.
        
        Args:
            model: The model to sample from.
            noise: The initial noise tensor.
            cond: conditional information.
            neg_cond: negative conditional information.
            steps: The number of steps to sample.
            rescale_t: The rescale factor for t.
            cfg_strength: The strength of classifier-free guidance.
            cfg_interval: The interval for classifier-free guidance.
            verbose: If True, show a progress bar.
            **kwargs: Additional arguments for model_inference.

        Returns:
            a dict containing the following
            - 'samples': the model samples.
            - 'pred_x_t': a list of prediction of x_t.
            - 'pred_x_0': a list of prediction of x_0.
        """

        return super().sample(is_structure,z0,z1,alpha_list, model, noise, cond_01, cond, steps, rescale_t, verbose, neg_cond=neg_cond, cfg_strength=cfg_strength, cfg_interval=cfg_interval, **kwargs)




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



def group_and_reorder_by_group_similarity(embed0, embed1, group_size=50):
    """
    对两个 embedding 按顺序划分成若干组，计算组之间的平均 embedding 之间的 similarity，
    用 Hungarian 算法对组进行匹配，然后 reorder 所有 token，保持不重复、每组完整、总长度一致。
    
    Args:
        embed0, embed1: [1, T, D] 形状的张量
        group_size: 每组的 token 数量（最后一组可小于该值）

    Returns:
        reordered0, reordered1: [1, T, D]，组匹配后重新排序的 embedding
    """
    device = embed0.device
    B, T, D = embed0.shape
    #pdb.set_trace()
    assert B == 1, "只支持 batch size = 1"
    embed0_flat = embed0[0]  # [T, D]
    embed1_flat = embed1[0]

    # === Step 1: 顺序分组 ===
    def split_into_groups(embed, group_size):
        T = embed.shape[0]
        indices = torch.arange(T, device=embed.device)
        groups = [indices[i:i+group_size] for i in range(0, T, group_size)]
        return groups  # list of [group_len] index tensors

    groups0 = split_into_groups(embed0_flat, group_size)
    groups1 = split_into_groups(embed1_flat, group_size)
    assert len(groups0) == len(groups1), "组数不一致（应使用相同方式分组）"
    num_groups = len(groups0)

    # === Step 2: 计算每个组的平均向量 ===
    means0 = torch.stack([embed0_flat[idx].mean(0) for idx in groups0])  # [G, D]
    means1 = torch.stack([embed1_flat[idx].mean(0) for idx in groups1])  # [G, D]

    # === Step 3: 组间 similarity 和 Hungarian 匹配 ===
    norm0 = F.normalize(means0, dim=1)
    norm1 = F.normalize(means1, dim=1)
    sim = norm0 @ norm1.T  # [G, G]
    cost = 1.0 - sim.cpu().numpy()
    row_ind, col_ind = linear_sum_assignment(cost)


    row_ind = torch.as_tensor(row_ind, device=device, dtype=torch.long)
    col_ind = torch.as_tensor(col_ind, device=device, dtype=torch.long)


    new_indices0 = torch.cat([groups0[i] for i in row_ind], dim=0)#torch.cat([groups0[row_ind[p]] for p in perm.tolist()], dim=0)#torch.cat([groups0[i] for i in row_ind], dim=0)
    new_indices1 = torch.cat([groups1[i] for i in col_ind], dim=0)#torch.cat([groups1[col_ind[p]] for p in perm.tolist()], dim=0)

    # 按共同顺序重排两侧（组内顺序不变）
    #new_indices0 = torch.cat([groups0[r] for r in row_ind], dim=0)#([groups0[row_ind[p]] for p in perm.tolist()], dim=0)

###
    reordered0 = embed0_flat.index_select(0, new_indices0).unsqueeze(0)  # [1, T, D]
    reordered1 = embed1_flat.index_select(0, new_indices1).unsqueeze(0)
###
    return reordered0, reordered1







def downsample_then_match_multi_group_new(
    slat0, slat1, grid_size=64, target_size=16, group_size=32, tau=0.6
):
    """
    新版本：返回的 idx_pairs 是 (voxel_id0, voxel_id1)，而不是 (token_id0, token_id1)
    这样可以正确映射到 16^3 空间的 KV cache
    """
    device = slat0.feats.device
    stride = grid_size // target_size

    def to_voxels(slat):
        coords16 = slat.coords // stride
        idx16 = (
            coords16[:, 0] * target_size * target_size
            + coords16[:, 1] * target_size
            + coords16[:, 2]
        )
        uniq, inv = torch.unique(idx16, return_inverse=True)
        pooled = torch.zeros(len(uniq), slat.feats.size(1), device=device, dtype=slat.feats.dtype)
        pooled.index_add_(0, inv, slat.feats)
        counts = torch.bincount(inv, minlength=len(uniq)).float().unsqueeze(-1).to(device)
        pooled = pooled / (counts + 1e-6)
        voxel2tokens = {u.item(): (idx16 == u).nonzero(as_tuple=True)[0].tolist() for u in uniq}
        # 构建 token -> voxel 映射
        token2voxel = {token_idx: voxel_id.item() for voxel_id in uniq for token_idx in voxel2tokens[voxel_id.item()]}
        return pooled, uniq, voxel2tokens, token2voxel

    f0, uniq0, vox2tok0, tok2vox0 = to_voxels(slat0)
    f1, uniq1, vox2tok1, tok2vox1 = to_voxels(slat1)
    if f0.numel() == 0 or f1.numel() == 0:
        return []

    # 分组
    group_stride = max(1, target_size // int(round((target_size**3 / group_size) ** (1 / 3))))
    gdim = target_size // group_stride

    def get_groups(uniq_idx, feats):
        coords = torch.stack([
            uniq_idx // (target_size*target_size),
            (uniq_idx // target_size) % target_size,
            uniq_idx % target_size
        ], dim=-1)
        gids = (coords // group_stride).long()
        gidx = gids[:, 0]*gdim*gdim + gids[:, 1]*gdim + gids[:, 2]
        groups = {}
        for i, gid in enumerate(gidx):
            gid = gid.item()
            if gid not in groups:
                groups[gid] = {"vox": [], "feats": []}
            groups[gid]["vox"].append(uniq_idx[i].item())
            groups[gid]["feats"].append(feats[i])
        for gid in groups:
            groups[gid]["feats"] = torch.stack(groups[gid]["feats"]).mean(0)
        return groups

    groups0 = get_groups(uniq0, f0)
    groups1 = get_groups(uniq1, f1)
    if not groups0 or not groups1:
        return []

    gids0, feats0 = zip(*[(gid, g["feats"]) for gid, g in groups0.items()])
    gids1, feats1 = zip(*[(gid, g["feats"]) for gid, g in groups1.items()])
    feats0, feats1 = torch.stack(feats0), torch.stack(feats1)

    # 匹配
    sim = F.normalize(feats0, dim=-1) @ F.normalize(feats1, dim=-1).T
    cost = (1 - sim).detach().cpu().numpy()
    row_ind, col_ind = linear_sum_assignment(cost)
    sim_vals = sim[row_ind, col_ind].detach().cpu().numpy()
    keep = sim_vals > tau
    row_ind, col_ind = row_ind[keep], col_ind[keep]

    idx_pairs = []
    for gi, gj in zip(row_ind, col_ind):
        voxels0 = groups0[gids0[gi]]["vox"]
        voxels1 = groups1[gids1[gj]]["vox"]
        L = min(len(voxels0), len(voxels1))
        for u0, u1 in zip(voxels0[:L], voxels1[:L]):
            tok0, tok1 = vox2tok0.get(u0, []), vox2tok1.get(u1, [])
            ltok = min(len(tok0), len(tok1))
            for t0, t1 in zip(tok0[:ltok], tok1[:ltok]):
                # 将 token 索引转换为 voxel 索引
                v0 = tok2vox0.get(t0, -1)
                v1 = tok2vox1.get(t1, -1)
                if v0 >= 0 and v1 >= 0:
                    idx_pairs.append((v0, v1))

    return idx_pairs


def reorder_kv_by_idx_sparse1_new(idx_pairs, model, anchor="0", grid_size=64, target_size=16):
    """
    新版本：处理 voxel ID 而不是 token ID，并改进去重逻辑
    idx_pairs 中的索引应该是 16^3 空间的 voxel ID（范围 0-4095）
    如果索引超出范围，会自动从 64^3 空间转换到 16^3 空间
    """
    if not idx_pairs:
        return
    device = next(model.parameters()).device
    idx0 = torch.tensor([p[0] for p in idx_pairs], dtype=torch.long, device=device)
    idx1 = torch.tensor([p[1] for p in idx_pairs], dtype=torch.long, device=device)

    for i, block in enumerate(model.blocks):
        attn = block.self_attn
        attn.layer_id = f"block_{i}"
        
        # 检查 kv_dict 中是否存在该 layer_id
        if attn.layer_id not in model.kv_dict_img0 or attn.layer_id not in model.kv_dict_img1:
            continue
            
        kv0 = model.kv_dict_img0[attn.layer_id]
        kv1 = model.kv_dict_img1[attn.layer_id]
        
        # 确保返回的是元组
        if not isinstance(kv0, tuple) or not isinstance(kv1, tuple):
            continue
            
        k0, v0 = kv0
        k1, v1 = kv1
        T = k0.size(1)

        stride = grid_size // target_size
     
        need_convert0 = (idx0 >= T) & (idx0 < grid_size**3)
        need_convert1 = (idx1 >= T) & (idx1 < grid_size**3)
        
        
        if need_convert0.any():
         
            x = idx0 // (grid_size * grid_size)
            y = (idx0 // grid_size) % grid_size
            z = idx0 % grid_size
          
            x16 = x // stride
            y16 = y // stride
            z16 = z // stride
           
            idx0_16 = x16 * target_size * target_size + y16 * target_size + z16
            
            idx0 = torch.where(need_convert0, idx0_16, idx0)
            
        if need_convert1.any():
        
            x = idx1 // (grid_size * grid_size)
            y = (idx1 // grid_size) % grid_size
            z = idx1 % grid_size
        
            x16 = x // stride
            y16 = y // stride
            z16 = z // stride
          
            idx1_16 = x16 * target_size * target_size + y16 * target_size + z16
         
            idx1 = torch.where(need_convert1, idx1_16, idx1)


        mask = (idx0 >= 0) & (idx0 < T) & (idx1 >= 0) & (idx1 < T)
        if not mask.any():
            continue
        idx0_safe, idx1_safe = idx0[mask], idx1[mask]


        seen = {}
        unique_pairs = []
        for j in range(len(idx0_safe)):
            vidx0_val, vidx1_val = idx0_safe[j].item(), idx1_safe[j].item()
            if vidx0_val not in seen:
                seen[vidx0_val] = vidx1_val
                unique_pairs.append((vidx0_val, vidx1_val))
        
        if not unique_pairs:
            continue
        uniq_idx0 = torch.tensor([p[0] for p in unique_pairs], dtype=torch.long, device=device)
        idx1_safe_new = torch.tensor([p[1] for p in unique_pairs], dtype=torch.long, device=device)

        if anchor == "0":
            k1[:, uniq_idx0, :, :] = k1[:, idx1_safe_new, :, :]
            v1[:, uniq_idx0, :, :] = v1[:, idx1_safe_new, :, :]
        else:  # anchor="1"
            k0[:, idx1_safe_new, :, :] = k0[:, uniq_idx0, :, :]
            v0[:, idx1_safe_new, :, :] = v0[:, uniq_idx0, :, :]

        model.kv_dict_img0[attn.layer_id] = (k0, v0)
        model.kv_dict_img1[attn.layer_id] = (k1, v1)