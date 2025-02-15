CODE_PATH=code
DATA_PATH=data
SAVE_PATH=models
GPU_DEVICE=${1}

BATCH_SIZE=${2}
NEG_SAMPLE_SIZE=${3}
POSGAMMA=${4}
NEGGAMMA=${5}
LR=${6}
EPOCH=${7}
HIDDEN_DIM=${8}
TARGET_DIM=${9}
SEED=${10}
PRETRAIN_PATH=${11}
DATASET=${12}
ENTITY_MUL=${13}
RELATION_MUL=${14}
OPTIMIZER=${15}
SCHEDULER=${16}
TEST_BATCH_SIZE=${17}

DATA_PATH=$DATA_PATH/$DATASET
SAVE_PATH=$SAVE_PATH/"$DATASET"_"$SEED"

CUDA_VISIBLE_DEVICES=$GPU_DEVICE python -u $CODE_PATH/Run.py -cuda \
    -seed $SEED \
    -data_path $DATA_PATH -save_path $SAVE_PATH -pretrain_path $PRETRAIN_PATH \
    -entity_mul $ENTITY_MUL -relation_mul $RELATION_MUL \
    -batch_size $BATCH_SIZE -negative_sample_size $NEG_SAMPLE_SIZE -pos_gamma $POSGAMMA -neg_gamma $NEGGAMMA\
    -lr $LR -epoch $EPOCH \
    -hidden_dim $HIDDEN_DIM -target_dim $TARGET_DIM \
    -optimizer $OPTIMIZER -scheduler $SCHEDULER \
    -test_batch_size $TEST_BATCH_SIZE \
    ${18} ${19} ${20} ${21} ${22} ${23} ${24} ${25} ${26} ${27}\
    ${28} ${29} ${30} ${31} ${32} ${33} ${34} ${35} ${36} ${37} ${38}\
    ${39} ${40} ${41} ${42} ${43} ${44} ${45} ${46} ${47} ${48} ${49}\
