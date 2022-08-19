from easynlp.appzoo import TextImageGeneration
from PIL import Image
import numpy as np
import albumentations
import torch
from io import BytesIO
import base64

def save_image(x):
    c,h,w = x.shape
    assert c==3
    x = ((x.detach().cpu().numpy().transpose(1,2,0)+1.0)*127.5).clip(0,255).astype(np.uint8)
    buffered = BytesIO()
    Image.fromarray(x).save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue())
    return str(img_str, 'utf-8')

def base64_to_image(imgbase64_str):
    image = Image.open(BytesIO(base64.urlsafe_b64decode(imgbase64_str)))
    return image


if __name__ == "__main__":
    # model = TextImageGeneration(pretrained_model_name_or_path='/home/yubin/EasyNLP-alibaba/pai-painter-base-zh')
    model = TextImageGeneration(pretrained_model_name_or_path='/home/yubin/EasyNLP-alibaba/tmp/finetune_model_MUGE')

    image = Image.open('/home/yubin/EasyNLP-alibaba/20220812104336.jpg').convert("RGB")
    image = np.array(image).astype(np.uint8)
    size = 256
    rescaler = albumentations.SmallestMaxSize(max_size = size)
    cropper = albumentations.CenterCrop(height=size,width=size)
    image = albumentations.Compose([rescaler, cropper])(image=image)["image"]
    Image.fromarray(image).save('./input.jpg')
    image = (image/127.5 - 1.0).astype(np.float32)
    image = np.expand_dims(image, axis=0)
    image = torch.from_numpy(image)
    image = image.permute(0, 3, 1, 2).to(memory_format=torch.contiguous_format)
    quant_z, indices = model.encode_to_z(image)
    recon_image = model.decode_to_img(indices, quant_z.shape)
    result = save_image(recon_image[0])
    newimage = base64_to_image(result)
    newimage.save('./input_recon.jpg')
    print('finish!')


