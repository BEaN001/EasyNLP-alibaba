import imp
import sys
import os

sys.path.append('./')

print('*'*50)
print('running local main...\n')
from easynlp.core import Trainer
# from easynlp.appzoo import get_application_evaluator
from easynlp.appzoo.sequence_classification.data import ClassificationDataset

from easynlp.appzoo import TextImageDataset
from easynlp.appzoo import TextImageGeneration
from easynlp.appzoo import TextImageGenerationEvaluator
from easynlp.appzoo import TextImageGenerationPredictor
from easynlp.utils import initialize_easynlp, get_args
from easynlp.utils.global_vars import parse_user_defined_parameters
from easynlp.core import PredictorManager
from easynlp.utils import get_pretrain_model_path


if __name__ == "__main__":
    print('log: starts to init...\n')
    # os.environ["NCCL_DEBUG_SUBSYS"] = "ALL"
    # os.environ["NCCL_DEBUG"] = "INFO"

    initialize_easynlp()
    args = get_args()
    # args.user_defined_parameters = {
    #     'size': 256,
    #     'text_len': 32,
    #     'img_len': 256,
    #     'img_vocab_size':16384,
    #     'pretrain_model_name_or_path': './pai-painter-base-zh'
    # }
    # python -m torch.distributed.launch $DISTRIBUTED_ARGS examples/text2image_generation/main.py \
    # --mode=train \
    # --worker_gpu=1 \
    # --tables=/data/yubindata/MUGE/MUGE_train_text_imgbase64.tsv,/data/yubindata/MUGE/MUGE_val_text_imgbase64.tsv \
    # --input_schema=idx:str:1,text:str:1,imgbase64:str:1 \
    # --first_sequence=text \
    # --second_sequence=imgbase64 \
    # --checkpoint_dir=./tmp/finetune_model \
    # --learning_rate=4e-5 \
    # --epoch_num=40 \
    # --random_seed=42 \
    # --logging_steps=100 \
    # --save_checkpoint_steps=1000 \
    # --sequence_length=288 \
    # --micro_batch_size=32 \
    # --app_name=text2image_generation \
    # --user_defined_parameters='
    #     pretrain_model_name_or_path=./pai-painter-base-zh
    #     size=256
    #     text_len=32
    #     img_len=256
    #     img_vocab_size=16384
    #   ' 
    args.user_defined_parameters = 'pretrain_model_name_or_path=./pai-painter-base-zh size=256 text_len=32 img_len=256 img_vocab_size=16384' 
    args.mode = 'train'
    args.worker_gpu = 1
    args.tables = '/data/yubindata/MUGE/MUGE_train_text_imgbase64.tsv,/data/yubindata/MUGE/MUGE_val_text_imgbase64.tsv'
    args.input_schema = 'idx:str:1,text:str:1,imgbase64:str:1'
    args.first_sequence = 'text'
    args.second_sequence = 'imgbase64'
    args.checkpoint_dir = './tmp/finetune_model'
    args.learning_rate = 4e-5
    args.epoch_num = 40
    args.random_seed = 42
    args.logging_steps = 100
    args.save_checkpoint_steps = 1000
    args.sequence_length = 288
    args.micro_batch_size = 32
    args.app_name = 'text2image_generation'

    print('log: starts to process user params...\n')
    user_defined_parameters = parse_user_defined_parameters(args.user_defined_parameters)
    if args.mode == "train" or not args.checkpoint_dir:
        args.pretrained_model_name_or_path = user_defined_parameters.get('pretrain_model_name_or_path', None)
    else:
        args.pretrained_model_name_or_path = args.checkpoint_dir

    pretrained_model_name_or_path = get_pretrain_model_path(args.pretrained_model_name_or_path)


    # if args.mode == "predict":
    #     predictor = TextImageGenerationPredictor(model_dir=args.checkpoint_dir, model_cls=TextImageGeneration,
    #                                    first_sequence=args.first_sequence, user_defined_parameters=user_defined_parameters)

    #     predictor_manager = PredictorManager(
    #         predictor=predictor,
    #         input_file=args.tables.split(",")[0],
    #         input_schema=args.input_schema,
    #         output_file=args.outputs,
    #         output_schema=args.output_schema,
    #         append_cols=args.append_cols,
    #         batch_size=args.micro_batch_size
    #     )
    #     predictor_manager.run()
    #     exit()


    print('log: starts to process dataset...\n')

    train_dataset = TextImageDataset(
        pretrained_model_name_or_path=pretrained_model_name_or_path,
        data_file=args.tables.split(",")[0],
        max_seq_length=args.sequence_length,
        input_schema=args.input_schema,
        first_sequence=args.first_sequence,
        second_sequence=args.second_sequence,
        user_defined_parameters=user_defined_parameters,
        is_training=True)

    valid_dataset = TextImageDataset(
        pretrained_model_name_or_path=pretrained_model_name_or_path,
        data_file=args.tables.split(",")[-1],
        max_seq_length=args.sequence_length,
        input_schema=args.input_schema,
        first_sequence=args.first_sequence,
        second_sequence=args.second_sequence,
        user_defined_parameters=user_defined_parameters,
        is_training=False)
    
    
    model = TextImageGeneration(pretrained_model_name_or_path=pretrained_model_name_or_path, user_defined_parameters=user_defined_parameters)
    evaluator = TextImageGenerationEvaluator(valid_dataset=valid_dataset, user_defined_parameters=user_defined_parameters)

    trainer = Trainer(model=model, train_dataset=train_dataset, user_defined_parameters=user_defined_parameters,
                      evaluator=evaluator)
    trainer.train()
