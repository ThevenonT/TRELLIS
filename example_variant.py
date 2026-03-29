import os
# os.environ['ATTN_BACKEND'] = 'sdpa'       # Safer default for wider GPU compatibility.
os.environ.setdefault('SPCONV_ALGO', 'native')
os.environ.setdefault('TRELLIS_DEVICE', 'cpu')

import imageio
import numpy as np
import open3d as o3d
import torch
from trellis.pipelines import TrellisTextTo3DPipeline
from trellis.utils import render_utils, postprocessing_utils

# Load a pipeline from a model folder or a Hugging Face model hub.
pipeline = TrellisTextTo3DPipeline.from_pretrained("microsoft/TRELLIS-text-xlarge")
if os.environ.get("TRELLIS_DEVICE", "cpu").lower() == "cuda" and torch.cuda.is_available():
    pipeline.cuda()
else:
    pipeline.cpu()

# Load mesh to make variants
base_mesh = o3d.io.read_triangle_mesh("assets/T.ply")

# Run the pipeline
outputs = pipeline.run_variant(
    base_mesh,
    "Rugged, metallic texture with orange and white paint finish, suggesting a durable, industrial feel.",
    seed=1,
    # Optional parameters
    # slat_sampler_params={
    #     "steps": 12,
    #     "cfg_strength": 7.5,
    # },
)
# outputs is a dictionary containing generated 3D assets in different formats:
# - outputs['gaussian']: a list of 3D Gaussians
# - outputs['radiance_field']: a list of radiance fields
# - outputs['mesh']: a list of meshes

# Render the outputs
video_gs = render_utils.render_video(outputs['gaussian'][0])['color']
video_mesh = render_utils.render_video(outputs['mesh'][0])['normal']
video = [np.concatenate([frame_gs, frame_mesh], axis=1) for frame_gs, frame_mesh in zip(video_gs, video_mesh)]
imageio.mimsave("sample_variant.mp4", video, fps=30)
