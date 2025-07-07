import torch
from diffusers.utils import export_to_video
from diffusers import AutoencoderKLWan, WanPipeline
from diffusers.schedulers.scheduling_unipc_multistep import UniPCMultistepScheduler
import gc
import time


def wan21_t2v_model(prompt:str, output_path="./video_output/wan2.1/"):
    
    vae = AutoencoderKLWan.from_pretrained('./models/wanai_t2v/', subfolder="vae", torch_dtype=torch.float32)
    flow_shift = 3.0 # 5.0 for 720P, 3.0 for 480P
    scheduler = UniPCMultistepScheduler(prediction_type='flow_prediction', use_flow_sigmas=True, num_train_timesteps=1000, flow_shift=flow_shift)
    pipe = WanPipeline.from_pretrained('./models/wanai_t2v/', vae=vae, torch_dtype=torch.bfloat16)
    pipe.scheduler = scheduler
    pipe.enable_model_cpu_offload()
    
    negative_prompt = "Bright tones, overexposed, static, blurred details, subtitles, style, works, paintings, images, static, overall gray, worst quality, low quality, JPEG compression residue, ugly, incomplete, extra fingers, poorly drawn hands, poorly drawn faces, deformed, disfigured, misshapen limbs, fused fingers, still picture, messy background, three legs, many people in the background, walking backwards"

    output = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        height=480,
        width=832,
        num_frames=33,
        guidance_scale=6.0,
        num_inference_steps=15
    ).frames[0]

    file_name=output_path+f"{time.strftime('%m%d-%H%M%S', time.localtime())}.mp4"
    export_to_video(output, file_name, fps=16)
    
    file_name=file_name.replace(".","",1)

    # release vram,gpu
    del pipe
    gc.collect()
    torch.cuda.empty_cache()

    return file_name


