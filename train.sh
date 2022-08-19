export CUDA_VISIBLE_DEVICES=0,1

MASTER_ADDR=localhost
MASTER_PORT=6007
GPUS_PER_NODE=2
NNODES=1
NODE_RANK=0

DISTRIBUTED_ARGS="--nproc_per_node $GPUS_PER_NODE --nnodes $NNODES --node_rank $NODE_RANK --master_addr $MASTER_ADDR --master_port $MASTER_PORT"

python -m torch.distributed.launch $DISTRIBUTED_ARGS examples/text2image_generation/main.py \
    --mode=train \
    --worker_gpu=1 \
    --tables=/data/yubindata/MUGE/MUGE_train_text_imgbase64.tsv,/data/yubindata/MUGE/MUGE_val_text_imgbase64.tsv \
    --input_schema=idx:str:1,text:str:1,imgbase64:str:1 \
    --first_sequence=text \
    --second_sequence=imgbase64 \
    --checkpoint_dir=./tmp/finetune_model \
    --learning_rate=4e-5 \
    --epoch_num=40 \
    --random_seed=42 \
    --logging_steps=100 \
    --save_checkpoint_steps=1000 \
    --sequence_length=288 \
    --micro_batch_size=32 \
    --app_name=text2image_generation \
    --user_defined_parameters='
        pretrain_model_name_or_path=./pai-painter-base-zh
        size=256
        text_len=32
        img_len=256
        img_vocab_size=16384
      ' 