from easynlp.utils import get_pretrain_model_path
from easynlp.appzoo import TextImageGenerationPredictor
from easynlp.core import PredictorManager
from easynlp.appzoo import TextImageGeneration
from PIL import Image
import base64
from io import BytesIO
def base64_to_image(imgbase64_str):
    image = Image.open(BytesIO(base64.urlsafe_b64decode(imgbase64_str)))
    return image
def format_input(self, inputs):
    """
    Preprocess single sentence data.
    """
    if type(inputs) != str and type(inputs) != list:
        raise RuntimeError("Input only supports strings or lists of strings")
    if type(inputs) == str:
        inputs = [inputs]
    return [{'first_sequence': input_sentence} for input_sentence in inputs]

if __name__ == "__main__":
    # user_defined_parameters = {
    #     'size': 256,
    #     'text_len': 32,
    #     'img_len': 256,
    #     'img_vocab_size':16384,
    #     'max_generated_num': 1
    # }
    user_defined_parameters = {}
    pretrained_model_name_or_path = get_pretrain_model_path('/home/yubin/EasyNLP-alibaba/tmp/finetune_model_MUGE')
    predictor = TextImageGenerationPredictor(model_dir=pretrained_model_name_or_path, model_cls=TextImageGeneration,
                                            user_defined_parameters=user_defined_parameters)
    # predictor = TextImageGenerationPredictor(model_dir=pretrained_model_name_or_path, model_cls=TextImageGeneration,
    #                                    first_sequence='text', user_defined_parameters=user_defined_parameters)

    # predictor_manager = PredictorManager(
    #     predictor=predictor,  
    #     input_file=args.tables.split(",")[0],
    #     input_schema=args.input_schema,
    #     output_file=args.outputs,
    #     output_schema=args.output_schema,
    #     append_cols=args.append_cols,
    #     batch_size=args.micro_batch_size
    # )
    # predictor_manager.run()
    data = ["粉色女性夏季性感内衣"]
    inputs = [{'idx': idx, 'first_sequence': data[idx]} for idx in range(len(data))]
    # model_inputs = predictor.preprocess(inputs)
    # model_outputs = predictor.predict(model_inputs)
    # results = predictor.postprocess(model_outputs)

    # results = predictor.run(inputs)

    model_inputs = predictor.preprocess(inputs)
    results = predictor.predict_postprocess(model_inputs)

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

    print('finish!')
