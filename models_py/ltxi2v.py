import torch
from diffusers import LTXImageToVideoPipeline
from diffusers.utils import export_to_video
from models_py.resizer import resize_img
import time
import gc



def LTX_i2v_model(prompt:str, input_image:str, output_path="./video_output/LTX/"):
    
    pipe = LTXImageToVideoPipeline.from_pretrained('./models/LTX', torch_dtype=torch.bfloat16)
    pipe.to("cuda")
    pipe.enable_model_cpu_offload()
    
    image = resize_img(input_img=input_image, max_size=704)
    negative_prompt = "worst quality, inconsistent motion, blurry, jittery, distorted"

    output = pipe(
        image=image,
        prompt=prompt,
        negative_prompt=negative_prompt,
        height=480,
        width=704,
        num_frames=33,
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


