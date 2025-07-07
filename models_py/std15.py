from diffusers import StableDiffusionPipeline
import torch
import time
import gc


def sd15model(prompt:str, output_path="./images_output/sd15/"):
    model_path = "./models/sd1.5"
    pipe = StableDiffusionPipeline.from_pretrained(model_path, torch_dtype=torch.float16).to("cuda")

    prompt = prompt

    image = pipe(
        prompt,
        num_inference_steps=35,
        guidance_scale= 7.5,
        generator=torch.Generator(device="cpu").manual_seed(int(time.time()))
    ).images[0]  
    
    file_name=output_path+f"{time.strftime('%m%d-%H%M%S', time.localtime())}.png"
    image.save(file_name)
    file_name=file_name.replace(".","",1)

    del pipe
    gc.collect()
    torch.cuda.empty_cache()

    return file_name

#實測不用5秒就好了