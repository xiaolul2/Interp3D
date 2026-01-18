import torch
import numpy as np
from tqdm import tqdm
import utils3d
from PIL import Image

from ..renderers import OctreeRenderer, GaussianRenderer, MeshRenderer
from ..representations import Octree, Gaussian, MeshExtractResult
from ..modules import sparse as sp
from .random_utils import sphere_hammersley_sequence
import pdb

def yaw_pitch_r_fov_to_extrinsics_intrinsics(yaws, pitchs, rs, fovs):
    is_list = isinstance(yaws, list)
    if not is_list:
        yaws = [yaws]
        pitchs = [pitchs]
    if not isinstance(rs, list):
        rs = [rs] * len(yaws)
    if not isinstance(fovs, list):
        fovs = [fovs] * len(yaws)
    extrinsics = []
    intrinsics = []
    for yaw, pitch, r, fov in zip(yaws, pitchs, rs, fovs):
        fov = torch.deg2rad(torch.tensor(float(fov))).cuda()
        yaw = torch.tensor(float(yaw)).cuda()
        pitch = torch.tensor(float(pitch)).cuda()
        orig = torch.tensor([
            torch.sin(yaw) * torch.cos(pitch),
            torch.cos(yaw) * torch.cos(pitch),
            torch.sin(pitch),
        ]).cuda() * r
        extr = utils3d.torch.extrinsics_look_at(orig, torch.tensor([0, 0, 0]).float().cuda(), torch.tensor([0, 0, 1]).float().cuda())
        intr = utils3d.torch.intrinsics_from_fov_xy(fov, fov)
        extrinsics.append(extr)
        intrinsics.append(intr)
    if not is_list:
        extrinsics = extrinsics[0]
        intrinsics = intrinsics[0]
    return extrinsics, intrinsics


def get_renderer(sample, **kwargs):
    if isinstance(sample, Octree): 
        renderer = OctreeRenderer()
        renderer.rendering_options.resolution = kwargs.get('resolution', 512)
        renderer.rendering_options.near = kwargs.get('near', 0.8)
        renderer.rendering_options.far = kwargs.get('far', 1.6)
        renderer.rendering_options.bg_color = kwargs.get('bg_color', (0, 0, 0))
        renderer.rendering_options.ssaa = kwargs.get('ssaa', 4)
        renderer.pipe.primitive = sample.primitive
    elif isinstance(sample, Gaussian):
        renderer = GaussianRenderer()
        renderer.rendering_options.resolution = kwargs.get('resolution', 512)
        renderer.rendering_options.near = kwargs.get('near', 0.8)
        renderer.rendering_options.far = kwargs.get('far', 1.6)
        renderer.rendering_options.bg_color = kwargs.get('bg_color', (0, 0, 0))
        renderer.rendering_options.ssaa = kwargs.get('ssaa', 1)
        renderer.pipe.kernel_size = kwargs.get('kernel_size', 0.1)
        renderer.pipe.use_mip_gaussian = True
    elif isinstance(sample, MeshExtractResult):
        renderer = MeshRenderer()
        renderer.rendering_options.resolution = kwargs.get('resolution', 512)
        renderer.rendering_options.near = kwargs.get('near', 1)
        renderer.rendering_options.far = kwargs.get('far', 100)
        renderer.rendering_options.ssaa = kwargs.get('ssaa', 4)
    else:
        raise ValueError(f'Unsupported sample type: {type(sample)}')
    return renderer


def render_frames(sample, extrinsics, intrinsics, options={}, colors_overwrite=None, verbose=True, **kwargs):
    #pdb.set_trace()
    renderer = get_renderer(sample, **options)
    rets = {}
    for j, (extr, intr) in tqdm(enumerate(zip(extrinsics, intrinsics)), desc='Rendering', disable=not verbose):
        if isinstance(sample, MeshExtractResult):
            res = renderer.render(sample, extr, intr)
            if 'normal' not in rets: rets['normal'] = []
            rets['normal'].append(np.clip(res['normal'].detach().cpu().numpy().transpose(1, 2, 0) * 255, 0, 255).astype(np.uint8))
        else:
            res = renderer.render(sample, extr, intr, colors_overwrite=colors_overwrite)
            if 'color' not in rets: rets['color'] = []
            if 'depth' not in rets: rets['depth'] = []
            rets['color'].append(np.clip(res['color'].detach().cpu().numpy().transpose(1, 2, 0) * 255, 0, 255).astype(np.uint8))
            if 'percent_depth' in res:
                rets['depth'].append(res['percent_depth'].detach().cpu().numpy())
            elif 'depth' in res:
                rets['depth'].append(res['depth'].detach().cpu().numpy())
            else:
                rets['depth'].append(None)
    return rets

def render_frames_rgba(sample, extrinsics, intrinsics, options={}, colors_overwrite=None, verbose=True, **kwargs):
    renderer = get_renderer(sample, **options)
    rets = {}

    for j, (extr, intr) in tqdm(enumerate(zip(extrinsics, intrinsics)), desc='Rendering', disable=not verbose):
        if isinstance(sample, MeshExtractResult):
            res = renderer.render(sample, extr, intr)
            if 'normal' not in rets:
                rets['normal'] = []
            rets['normal'].append(
                np.clip(res['normal'].detach().cpu().numpy().transpose(1, 2, 0) * 255, 0, 255).astype(np.uint8)
            )
        else:
            # --- 渲染黑底 ---
            opts_black = options.copy()
            opts_black["bg_color"] = (0.0, 0.0, 0.0)
            res_black = renderer.render(sample, extr, intr, colors_overwrite=colors_overwrite)

            # --- 渲染白底 ---
            opts_white = options.copy()
            opts_white["bg_color"] = (1.0, 1.0, 1.0)
            res_white = get_renderer(sample, **opts_white).render(sample, extr, intr, colors_overwrite=colors_overwrite)

            rgb_black = res_black['color'].detach().cpu().numpy().transpose(1, 2, 0)  # float [0,1]
            rgb_white = res_white['color'].detach().cpu().numpy().transpose(1, 2, 0)

            # --- 计算 alpha 和前景色 ---
            alpha = 1.0 - (rgb_white - rgb_black).mean(axis=-1, keepdims=True)  # [H,W,1]
            alpha = np.clip(alpha, 0.0, 1.0)

            rgba = np.concatenate([rgb_black, alpha], axis=-1)
            rgba = (rgba * 255).astype(np.uint8)

            if 'color' not in rets:
                rets['color'] = []
            if 'depth' not in rets:
                rets['depth'] = []

            rets['color'].append(rgba)

            if 'percent_depth' in res_black:
                rets['depth'].append(res_black['percent_depth'].detach().cpu().numpy())
            elif 'depth' in res_black:
                rets['depth'].append(res_black['depth'].detach().cpu().numpy())
            else:
                rets['depth'].append(None)

    return rets


def render_video(sample, resolution=512, bg_color=(0, 0, 0), num_frames=60, r=2, fov=40, **kwargs):
    yaws = torch.linspace(0, 2 * 3.1415, num_frames)
    pitch = 0.25 + 0.5 * torch.sin(torch.linspace(0, 2 * 3.1415, num_frames))
    yaws = yaws.tolist()
    pitch = pitch.tolist()
    extrinsics, intrinsics = yaw_pitch_r_fov_to_extrinsics_intrinsics(yaws, pitch, r, fov)
    #pdb.set_trace()
    return render_frames(sample, extrinsics, intrinsics, {'resolution': resolution, 'bg_color': bg_color}, **kwargs)


def render_multiview(sample, resolution=512, nviews=30):
    r = 2
    fov = 40
    cams = [sphere_hammersley_sequence(i, nviews) for i in range(nviews)]
    yaws = [cam[0] for cam in cams]
    pitchs = [cam[1] for cam in cams]
    extrinsics, intrinsics = yaw_pitch_r_fov_to_extrinsics_intrinsics(yaws, pitchs, r, fov)
    res = render_frames(sample, extrinsics, intrinsics, {'resolution': resolution, 'bg_color': (0, 0, 0)})
    return res['color'], extrinsics, intrinsics


def render_multiview_rgba(sample, resolution=512, nviews=30):
    r = 2
    fov = 40
    cams = [sphere_hammersley_sequence(i, nviews) for i in range(nviews)]
    yaws = [cam[0] for cam in cams]
    pitchs = [cam[1] for cam in cams]
    extrinsics, intrinsics = yaw_pitch_r_fov_to_extrinsics_intrinsics(yaws, pitchs, r, fov)
    res = render_frames_rgba(
        sample, extrinsics, intrinsics, {'resolution': resolution, 'bg_color': (0, 0, 0)}
    )
    return res['color'], extrinsics, intrinsics




def render_multiview_mesh(sample, resolution=512, nviews=30):
    r = 2
    fov = 40
    cams = [sphere_hammersley_sequence(i, nviews) for i in range(nviews)]
    yaws = [cam[0] for cam in cams]
    pitchs = [cam[1] for cam in cams]
    extrinsics, intrinsics = yaw_pitch_r_fov_to_extrinsics_intrinsics(
        yaws, pitchs, r, fov
    )

    # 渲染时必须传 bg_color
    res = render_frames(
        sample,
        extrinsics,
        intrinsics,
        {'resolution': resolution, 'bg_color': (0, 0, 0)}
    )

    normals = res['normal']   # list of (H,W,3)
    alphas = res.get('alpha', None)  # list of (H,W,1)，float [0,1] or [0,255]

    rgba_list = []
    for idx, nmap in enumerate(normals):
        if nmap.shape[-1] == 4:  # 已经是 RGBA
            rgba_list.append(nmap)
            continue

        # alpha 通道
        if alphas is not None:
            alpha = alphas[idx]
            if alpha.max() <= 1.0:   # [0,1] -> [0,255]
                alpha = (alpha * 255).astype(np.uint8)
            else:
                alpha = alpha.astype(np.uint8)
        else:
            mask = (nmap.sum(axis=-1, keepdims=True) > 0).astype(np.uint8)
            alpha = mask * 255

        # ---- Premultiplied alpha 防黑毛刺 ----
        nmap = nmap.astype(np.float32)
        alpha_f = alpha.astype(np.float32) / 255.0
        nmap = (nmap * alpha_f).clip(0, 255).astype(np.uint8)

        rgba = np.concatenate([nmap, alpha], axis=-1)
        rgba_list.append(rgba)

    return rgba_list, extrinsics, intrinsics

def fix_black_halo(rgb, alpha):
    rgb = rgb.astype(np.float32)  # (H,W,3)
    alpha_f = alpha.astype(np.float32) / 255.0  # (H,W,1)

    # 避免除零
    alpha_safe = np.where(alpha_f > 1e-6, alpha_f, 1.0)

    # 直接用广播修复，不要 repeat
    rgb = rgb / alpha_safe  

    # 背景直接清零
    rgb[alpha_f.squeeze(-1) <= 1e-6] = 0  

    return np.clip(rgb, 0, 255).astype(np.uint8)

def render_snapshot(samples, resolution=512, bg_color=(0, 0, 0), offset=(-16 / 180 * np.pi, 20 / 180 * np.pi), r=10, fov=8, **kwargs):
    yaw = [0, np.pi/2, np.pi, 3*np.pi/2]
    yaw_offset = offset[0]
    yaw = [y + yaw_offset for y in yaw]
    pitch = [offset[1] for _ in range(4)]
    extrinsics, intrinsics = yaw_pitch_r_fov_to_extrinsics_intrinsics(yaw, pitch, r, fov)
    return render_frames(samples, extrinsics, intrinsics, {'resolution': resolution, 'bg_color': bg_color}, **kwargs)
