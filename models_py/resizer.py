from PIL import Image
from diffusers.utils import load_image

def resize_img(input_img:str, max_size:int):
    image = load_image(input_img)
    width, height = image.size
    scale = max_size / max(width, height)

    new_width = int(width * scale)
    new_height = int(height * scale)

    resized_image = image.resize((new_width, new_height), Image.LANCZOS)
    return resized_image
