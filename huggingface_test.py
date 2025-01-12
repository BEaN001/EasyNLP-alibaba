import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# load model and tokenizer
model_id = "distilbert-base-uncased-finetuned-sst-2-english"
model = AutoModelForSequenceClassification.from_pretrained(model_id).cuda()
tokenizer = AutoTokenizer.from_pretrained(model_id)
dummy_model_input = tokenizer("This is a sample", return_tensors="pt")

example_out = model(tuple(dummy_model_input.values())[0].cuda(), tuple(dummy_model_input.values())[1].cuda())
print(example_out)


# export
torch.onnx.export(
    model, 
    # tuple(dummy_model_input.values()),
    (tuple(dummy_model_input.values())[0].cuda(), tuple(dummy_model_input.values())[1].cuda()),
    f="torch-model.onnx",  
    input_names=['input_ids', 'attention_mask'], 
    output_names=['logits'], 
    dynamic_axes={'input_ids': {0: 'batch_size', 1: 'sequence'}, 
                  'attention_mask': {0: 'batch_size', 1: 'sequence'}, 
                  'logits': {0: 'batch_size', 1: 'sequence'}}, 
    do_constant_folding=True, 
    opset_version=13, 
)