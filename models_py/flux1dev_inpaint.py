from models_py.controlnet_img import get_depth_map
import torch
from diffusers import FluxControlNetPipeline, FluxControlNetModel
import numpy as np
import gc
import time

def flux_dev_inpaint_model(prompt:str, input_image:str, max_size=512,control_mode=2,controlnet_conditioning_scale=0.6 , output_path="./images_output/flux1dev/"):
    controlnet = FluxControlNetModel.from_pretrained('./models/flux_controlnet', torch_dtype=torch.bfloat16)
    pipe = FluxControlNetPipeline.from_pretrained('./models/flux1dev', controlnet=controlnet, torch_dtype=torch.bfloat16).to('cuda')
    pipe.enable_model_cpu_offload()
    depth_image, width, height = get_depth_map(input_image, max_size=max_size)

    image = pipe(
        prompt=prompt,
        control_image=depth_image,
        width=width,
        height=height,
        num_inference_steps=15,
        control_mode=control_mode,
        controlnet_conditioning_scale=controlnet_conditioning_scale,
        generator=torch.Generator(device="cpu").manual_seed(int(time.time()))
    ).images[0]

    file_name=output_path+f"{time.strftime('%m%d-%H%M%S', time.localtime())}.png"
    image.save(file_name)
    file_name=file_name.replace(".","",1)

    # release vram,gpu
    del pipe
    gc.collect()
    torch.cuda.empty_cache()

    return file_name



