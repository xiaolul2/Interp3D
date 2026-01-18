import os
import glob
import argparse
import torch
import imageio
from PIL import Image
from tqdm import tqdm

from trellis.pipelines import TrellisImageTo3DPipeline
from trellis.utils import render_utils, postprocessing_utils


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='TRELLIS Interpolation Example')
    parser.add_argument('--exp_id', type=str, default='./case_alpha',
                        help='Experiment ID or path to cases directory')
    parser.add_argument('--seed', type=int, default=1,
                        help='Random seed for generation')
    parser.add_argument('--num_search_iterations', type=int, default=7,
                        help='Number of search iterations')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Output directory (default: ./output/{exp_id})')
    parser.add_argument('--cases_root', type=str, default=None,
                        help='Root directory for input cases (default: ./{exp_id})')
    return parser.parse_args()


def get_config(args):
    """Create config dictionary from arguments."""
    # Extract exp_id from path if it's a path
    if os.path.isdir(args.exp_id):
        exp_id = os.path.basename(os.path.normpath(args.exp_id))
        cases_root = args.exp_id
    else:
        exp_id = args.exp_id
        cases_root = args.cases_root if args.cases_root else f"./{exp_id}"
    
    out_dir = args.output_dir if args.output_dir else f"./output/{exp_id}"
    
    config = {
        'exp_id': exp_id,
        'cases_root': cases_root,
        'out_dir': out_dir,
        'model_path': args.model_path,
        'seed': args.seed,
        'num_search_iterations': args.num_search_iterations,
        'device': torch.device("cuda" if torch.cuda.is_available() else "cpu")
    }
    return config


def main():
    # Parse arguments and create config
    args = parse_args()
    config = get_config(args)
    
    # Create output directory
    os.makedirs(config['out_dir'], exist_ok=True)


    # ===== 收集所有 case =====
    all_cases = {}
    for path in glob.glob(os.path.join(config['cases_root'], "*.png")):
        name = os.path.basename(path)
        base = name[:-5]
        all_cases.setdefault(base, []).append(path)
    all_cases = {k: sorted(v) for k, v in all_cases.items() if len(v) >= 2}

    video_dir = os.path.join(config['out_dir'], "videos")
    os.makedirs(video_dir, exist_ok=True)
    visual_image_dir = os.path.join(config['out_dir'], "mv_images")
    os.makedirs(visual_image_dir, exist_ok=True)
    glb_dir = os.path.join(config['out_dir'], "glb")
    os.makedirs(glb_dir, exist_ok=True)

    for case_name, paths in tqdm(all_cases.items(), desc="Processing cases"):
        image1, image2 = Image.open(paths[0]), Image.open(paths[1])
        pipeline = TrellisImageTo3DPipeline.from_pretrained("microsoft/TRELLIS-image-large")
        pipeline.cuda()
        outputs, _ = pipeline.run_interpolation_morphing(
            image1, image2, seed=config['seed'], num_search_iterations=config['num_search_iterations']
        )

        # 渲染多视角图像和视频
        for i in range(len(outputs)):
            video = render_utils.render_video(outputs[i]['gaussian'][0])['color']
            imageio.mimsave(f"{video_dir}/{case_name}_{i}.mp4", video, fps=30)

            img_list = render_utils.render_multiview_rgba(outputs[i]['gaussian'][0], nviews=64)[0]
            png_dir = os.path.join(visual_image_dir, case_name)
            os.makedirs(png_dir, exist_ok=True)
            for j, img in enumerate(img_list):
                imageio.imwrite(os.path.join(png_dir, f"seq_{i}_view_{j}.png"), img)
            
            ### GLB files can be extracted from the outputs
            glb = postprocessing_utils.to_glb(
                outputs[i]['gaussian'][0],
                outputs[i]['mesh'][0],
                # Optional parameters
                simplify=0.95,          # Ratio of triangles to remove in the simplification process
                texture_size=1024,      # Size of the texture used for the GLB
            )
            glb.export(f"{glb_dir}/{case_name}_{i}.glb")


if __name__ == '__main__':
    main()