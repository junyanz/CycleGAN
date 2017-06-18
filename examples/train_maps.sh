DB_NAME='maps'
GPU_ID=1
DISPLAY_ID=1
NET_G=resnet_6blocks
NET_D=basic
MODEL=cycle_gan
SAVE_EPOCH=5
ALIGN_DATA=0
LAMBDA=10
NF=64


EXPR_NAME=${DB_NAME}_${MODEL}_${LAMBDA}

CHECKPOINT_DIR=./checkpoints/
LOG_FILE=${CHECKPOINT_DIR}${EXPR_NAME}/log.txt
mkdir -p ${CHECKPOINT_DIR}${EXPR_NAME}

DATA_ROOT=./datasets/$DB_NAME align_data=$ALIGN_DATA use_lsgan=1 \
which_direction='AtoB' display_plot=$PLOT pool_size=50 niter=100 niter_decay=100 \
which_model_netG=$NET_G which_model_netD=$NET_D model=$MODEL lr=0.0002 print_freq=200 lambda_A=$LAMBDA lambda_B=$LAMBDA \
loadSize=143 fineSize=128 gpu=$GPU_ID display_winsize=128 \
name=$EXPR_NAME flip=1 save_epoch_freq=$SAVE_EPOCH \
continue_train=0 display_id=$DISPLAY_ID \
checkpoints_dir=$CHECKPOINT_DIR\
 th train.lua | tee -a $LOG_FILE
