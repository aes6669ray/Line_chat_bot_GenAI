import torch
from diffusers import FluxImg2ImgPipeline
from diffusers.utils import load_image
import gc
import time


def flux_schnell_inpaint_model(prompt:str, input_image:str, output_path="./images_output/flux1schnell/"):
    model_path = "./models/flux1schnell"
    pipe = FluxImg2ImgPipeline.from_pretrained(model_path, torch_dtype=torch.bfloat16)
    pipe.enable_model_cpu_offload() #save some VRAM by offloading the model to CPU. Remove this if you have enough GPU power
    init_image = load_image(input_image)

    image = pipe(
        prompt=prompt,
        image=init_image,
        guidance_scale=9,
        num_inference_steps=10,
        strength=0.88,
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

