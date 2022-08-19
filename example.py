from easynlp.pipelines import pipeline
from PIL import Image
import base64
from io import BytesIO

def base64_to_image(imgbase64_str):
    image = Image.open(BytesIO(base64.urlsafe_b64decode(imgbase64_str)))
    return image

data = ['远处的雪山，表面覆盖着厚厚的积雪']
generator = pipeline('text2image_generation')

results = generator(data)

for text, result in zip(data, results):
    imgbase64_str_list = result['gen_imgbase64']
    imgpath_list = []
    for base64_idx in range(len(imgbase64_str_list)):
        imgbase64_str = imgbase64_str_list[base64_idx]
        image = base64_to_image(imgbase64_str)
        imgpath = '{}_{}.png'.format(text, base64_idx)
        image.save(imgpath)
        imgpath_list.append(imgpath)
    print ('text: {}, save generated image: {}'.format(text, imgpath_list))
-m torch.distributed.launch $DISTRIBUTED_ARGS
python examples/text2image_generation/main.py --mode=train --worker_gpu=2 --tables=/data/yubindata/MUGE/MUGE_train_text_imgbase64.tsv,/data/yubindata/MUGE/MUGE_val_text_imgbase64.tsv --input_schema=idx:str:1,text:str:1,imgbase64:str:1 --first_sequence=text --second_sequence=imgbase64 --checkpoint_dir=./tmp/finetune_model --learning_rate=4e-5 --epoch_num=40 --random_seed=42 --logging_steps=100 --save_checkpoint_steps=1000 --sequence_length=288 --micro_batch_size=16 --app_name=text2image_generation --user_defined_parameters='pretrain_model_name_or_path=./pai-painter-large-zh size=256 text_len=32 img_len=256 img_vocab_size=16384'