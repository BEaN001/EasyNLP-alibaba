# from PIL import Image
# import base64
# from io import BytesIO
# from easynlp.pipelines import pipeline


# # 直接构建pipeline
# default_ecommercial_pipeline = pipeline("pai-painter-commercial-base-zh")

# # 模型预测
# data = ["宽松T恤"]
# results = default_ecommercial_pipeline(data)  # results的每一条是生成图像的base64编码

# # base64转换为图像
# def base64_to_image(imgbase64_str):
#     image = Image.open(BytesIO(base64.urlsafe_b64decode(imgbase64_str)))
#     return image

# # 保存以文本命名的图像
# for text, result in zip(data, results):
#     imgpath = '{}.png'.format(text)
#     imgbase64_str = result['gen_imgbase64']
#     image = base64_to_image(imgbase64_str)
#     image.save(imgpath)
#     print('text: {}, save generated image: {}'.format(text, imgpath))

from easynlp.pipelines import pipeline
from PIL import Image
import base64
from io import BytesIO
def base64_to_image(imgbase64_str):
    image = Image.open(BytesIO(base64.urlsafe_b64decode(imgbase64_str)))
    return image

#data = ["一只俏皮的狗正跑过草地", "一片水域的景色以日落为背景"]
# data = ["一只猫在床上趴着"]
#data = [ "遥望吴山为谁好，忽闻楚些令人伤。","遥望吴山为谁好，忽闻楚些令人伤。","遥望吴山为谁好，忽闻楚些令人伤。","遥望吴山为谁好，忽闻楚些令人伤。","遥望吴山为谁好，忽闻楚些令人伤。",]
data = ["女童套头毛衣打底衫秋冬针织衫童装儿童内搭上衣"]
# data = [ "见说春风偏有贺，露花千朵照庭闱。"]
# generator = pipeline('text2image_generation', model_path="./pai-painter-commercial-base-zh")
# generator = pipeline('text2image_generation', model_path="./pai-painter-painting-base-zh")
# generator = pipeline('text2image_generation', model_path="./pai-painter-scenery-base-zh")
# generator = pipeline('text2image_generation', model_path="./pai-painter-large-zh")
# generator = pipeline('text2image_generation', model_path="./pai-painter-base-zh")
# generator = pipeline("pai-painter-painting-base-zh")
generator = pipeline('text2image_generation', model_path="./tmp/finetune_model")

results = generator(data)

for text, result in zip(data, results):
    imgbase64_str_list = result['gen_imgbase64']
    imgpath_list = []
    for base64_idx in range(len(imgbase64_str_list)):
        imgbase64_str = imgbase64_str_list[base64_idx]
        image = base64_to_image(imgbase64_str)
        imgpath = '{}_{}.png'.format(text, base64_idx)
        import os
        while os.path.exists(imgpath):
            base64_idx += 1
            imgpath = '{}_{}.png'.format(text, base64_idx)
        image.save(imgpath)
        imgpath_list.append(imgpath)
    print ('text: {}, save generated image: {}'.format(text, imgpath_list))