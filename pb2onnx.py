from easynlp.utils import get_pretrain_model_path
from easynlp.appzoo import TextImageGenerationPredictor
from easynlp.core import PredictorManager
from easynlp.appzoo import TextImageGeneration
from PIL import Image
import base64
from io import BytesIO
import torch

# import torch
# from transformers import AutoModelForSequenceClassification, AutoTokenizer

# # load model and tokenizer
# model_id = "distilbert-base-uncased-finetuned-sst-2-english"
# model = AutoModelForSequenceClassification.from_pretrained(model_id).cuda()
# tokenizer = AutoTokenizer.from_pretrained(model_id)
# dummy_model_input = tokenizer("This is a sample", return_tensors="pt")

# example_out = model(tuple(dummy_model_input.values())[0].cuda(), tuple(dummy_model_input.values())[1].cuda())
# print(example_out)


# # export
# torch.onnx.export(
#     model, 
#     # tuple(dummy_model_input.values()),
#     (tuple(dummy_model_input.values())[0].cuda(), tuple(dummy_model_input.values())[1].cuda()),
#     f="torch-model.onnx",  
#     input_names=['input_ids', 'attention_mask'], 
#     output_names=['logits'], 
#     dynamic_axes={'input_ids': {0: 'batch_size', 1: 'sequence'}, 
#                   'attention_mask': {0: 'batch_size', 1: 'sequence'}, 
#                   'logits': {0: 'batch_size', 1: 'sequence'}}, 
#     do_constant_folding=True, 
#     opset_version=13, 
# )

if __name__ == "__main__":

    user_defined_parameters = {'max_generated_num': 1}
    pretrained_model_name_or_path = get_pretrain_model_path('/home/yubin/EasyNLP-alibaba/tmp/finetune_model_MUGE')
    predictor = TextImageGenerationPredictor(model_dir=pretrained_model_name_or_path, model_cls=TextImageGeneration,
                                            user_defined_parameters=user_defined_parameters)
    model = predictor.model

    data = ["黄色秋冬针织衫童装儿童内搭上衣"]
    inputs = [{'idx': idx, 'first_sequence': data[idx]} for idx in range(len(data))]   

    model_inputs = predictor.preprocess(inputs)

    idx = model_inputs["idx"]
    text_ids = torch.LongTensor(model_inputs['input_ids']).cuda()

    bs = len(idx)
    cshape = torch.tensor([bs, 256, 16, 16])

    example_output = model(text_ids, cshape)
    print(text_ids.shape, cshape.shape)
    print(example_output.shape)

    torch.onnx.export(model, 
                    (text_ids, cshape),
                    f="text2imageMUGE.onnx", 
                    # opset_version=11,
                    input_names=['text_ids', 'cshape'],
                    output_names=['gen_imgs'],
                    dynamic_axes={'text_ids': {0: 'batch_size'},
                                # "cshape": {0: "batch_size"},
                                'gen_imgs': {0: 'batch_size'}},
                    do_constant_folding=True, 
                    opset_version=13, 
                    )

# if __name__ == "__main__":

#     user_defined_parameters = {'max_generated_num': 1}
#     pretrained_model_name_or_path = get_pretrain_model_path('/home/yubin/EasyNLP-alibaba/tmp/finetune_model_MUGE')
#     predictor = TextImageGenerationPredictor(model_dir=pretrained_model_name_or_path, model_cls=TextImageGeneration,
#                                             user_defined_parameters=user_defined_parameters)
#     # model = predictor.model

#     model = TextImageGeneration(pretrained_model_name_or_path='/home/yubin/EasyNLP-alibaba/tmp/finetune_model_MUGE').cuda()
#     model.eval()

#     data = ["黄色秋冬针织衫童装儿童内搭上衣"]
#     inputs = [{'idx': idx, 'first_sequence': data[idx]} for idx in range(len(data))]   

#     model_inputs = predictor.preprocess(inputs)

#     idx = model_inputs["idx"]
#     text_ids = torch.LongTensor(model_inputs['input_ids']).cuda()

#     bs = len(idx)
#     cshape = torch.tensor([bs, 256, 16, 16])

#     example_output = model(text_ids, cshape)
#     print(text_ids.shape, cshape.shape)
#     print(example_output.shape)

#     torch.onnx.export(model, 
#                     (text_ids, cshape),
#                     f="text2imageMUGE.onnx", 
#                     # opset_version=11,
#                     input_names=['text_ids', 'cshape'],
#                     output_names=['gen_imgs'],
#                     dynamic_axes={'text_ids': {0: 'batch_size'},
#                                 # "cshape": {0: "batch_size"},
#                                 'gen_imgs': {0: 'batch_size'}},
#                     do_constant_folding=True, 
#                     opset_version=13, 
#                     )


    # # Export the trained model to ONNX
    # dummy_input1 = torch.tensor([[24326, 22066, 21288, 17484, 23535, 21686, 22520, 21381, 22547, 17420,
    #         21381, 17463, 19406, 17061, 22516, 16384, 16384, 16384, 16384, 16384,
    #         16384, 16384, 16384, 16384, 16384, 16384, 16384, 16384, 16384, 16384,
    #         16384, 16384]]).cuda()
    # dummy_input2 = torch.tensor([  1, 256,  16,  16]).cuda()

    # example_output = predictor.model(dummy_input1, dummy_input2)
    # print(example_output.shape)

    # torch.onnx.export(predictor.model, 
    #                 (dummy_input1, dummy_input2),
    #                 "output/text2image.onnx", 
    #                 # opset_version=11,
    #                 input_names=["text_ids", "cshape"],
    #                 output_names=["gen_imgs"],
    #                 dynamic_axes={"text_ids": {0: "batch_size"},
    #                             "cshape": {0: "batch_size"},
    #                             "gen_imgs": {0: "batch_size"}},
    #                 do_constant_folding=True, 
    #                 opset_version=13, 
    #                 )
