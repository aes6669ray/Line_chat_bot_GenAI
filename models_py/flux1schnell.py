import torch
from diffusers import FluxPipeline
import gc
import time

def flux_schnell_model(prompt:str, output_path="./images_output/flux1schnell/"):
    model_path="./models/flux1schnell"
    pipe = FluxPipeline.from_pretrained(model_path, torch_dtype=torch.bfloat16)
    pipe.enable_model_cpu_offload() #save some VRAM by offloading the model to CPU. Remove this if you have enough GPU power

 

    image = pipe(
        prompt=prompt,
        guidance_scale=0.0,
        num_inference_steps=4,
        max_sequence_length=256,
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

##實測秒數大約在18~24秒
