#! /bin/sh

TRAIN_PATH=samples/sample1/train.tsv
DEV_PATH=samples/sample1/validation.tsv
EXPT_DIR=sample1_checkpoints

# set values
EMB_SIZE=16
H_SIZE=512
N_LAYERS=1
CELL='gru'
EPOCH=50
PRINT_EVERY=20
TF=0.5
OPTIM='adam'
LR=0.001
BATCH_SIZE=1
#LOAD_CHECKPOINT='acc_1.00_seq_acc_1.00_target_acc_1.00_nll_loss_0.00_attn_loss_0.00_s4640'


## Start training
echo "Train model on sample data"
python3 train_model.py --train $TRAIN_PATH --output_dir $EXPT_DIR --print_every $PRINT_EVERY --embedding_size $EMB_SIZE --hidden_size $H_SIZE --rnn_cell $CELL --n_layers $N_LAYERS --batch_size $BATCH_SIZE --epoch $EPOCH --print_every $PRINT_EVERY --attention 'pre-rnn' --attention_method 'mlp' --optim $OPTIM --lr $LR --full_focus --use_attention_loss

