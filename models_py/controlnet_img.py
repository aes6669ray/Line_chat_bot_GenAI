from transformers import pipeline
from PIL import Image
from diffusers.utils import load_image
import numpy as np


def get_depth_map(input_img, max_size=512,method="depth" , preview_control_image=False):

    image = load_image(input_img)
    width, height = image.size
    scale = max_size / max(width, height)

    new_width = int(width * scale)
    new_height = int(height * scale)

    image = image.resize((new_width, new_height), Image.LANCZOS)
    
    image = pipeline("depth-estimation")(image)[method]
    image = np.array(image)
    image = image[:, :, None]
    image = np.concatenate([image, image, image], axis=2)
    control_image = Image.fromarray(image)
    
    ### This part is for sd1.5 StableDiffusionControlNetImg2ImgPipeline, see more:https://huggingface.co/docs/diffusers/using-diffusers/controlnet
    # import torch
    # detected_map = torch.from_numpy(image).float() / 255.0
    # depth_map = detected_map.permute(2, 0, 1).unsqueeze(0).half().to("cuda")

    # 儲存depth_image
    if preview_control_image:
        control_image.save("control.png")

    return control_image, new_width, new_height