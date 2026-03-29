import os
# os.environ['ATTN_BACKEND'] = 'sdpa'       # Safer default for wider GPU compatibility.
os.environ.setdefault('SPCONV_ALGO', 'native')
os.environ.setdefault('TRELLIS_DEVICE', 'cpu')

import numpy as np
import imageio
import torch
from PIL import Image
from trellis.pipelines import TrellisImageTo3DPipeline
from trellis.utils import render_utils

# Load a pipeline from a model folder or a Hugging Face model hub.
pipeline = TrellisImageTo3DPipeline.from_pretrained("microsoft/TRELLIS-image-large")
if os.environ.get("TRELLIS_DEVICE", "cpu").lower() == "cuda" and torch.cuda.is_available():
    pipeline.cuda()
else:
    pipeline.cpu()

# Load an image
images = [
    Image.open("assets/example_multi_image/character_1.png"),
    Image.open("assets/example_multi_image/character_2.png"),
    Image.open("assets/example_multi_image/character_3.png"),
]

# Run the pipeline
outputs = pipeline.run_multi_image(
    images,
    seed=1,
    # Optional parameters
    sparse_structure_sampler_params={
        "steps": 12,
        "cfg_strength": 7.5,
    },
    slat_sampler_params={
        "steps": 12,
        "cfg_strength": 3,
    },
)
# outputs is a dictionary containing generated 3D assets in different formats:
# - outputs['gaussian']: a list of 3D Gaussians
# - outputs['radiance_field']: a list of radiance fields
# - outputs['mesh']: a list of meshes

video_gs = render_utils.render_video(outputs['gaussian'][0])['color']
video_mesh = render_utils.render_video(outputs['mesh'][0])['normal']
video = [np.concatenate([frame_gs, frame_mesh], axis=1) for frame_gs, frame_mesh in zip(video_gs, video_mesh)]
imageio.mimsave("sample_multi.mp4", video, fps=30)
