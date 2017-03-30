DB_NAME=cityscapes
# EXPR_NAME=${DB_NAME}_
# EXPR_NAMES=('resnet_unet128_basic_64_pix2pix')  #'resnet_unet128_basic_64_content_gan_pixel_1'
# EXPR_NAMES=('resnet_unet128_basic_64_A10_B10')
# S=('simple_pix2pix_10')

EXPR_NAMES=('gan_one_cycle_resnet_unet128_basic_64_10_backward')

# EXPR_NAMES=('gan_cycle_resnet_unet128_basic_64_10')
# EXPR_NAMES=('resnet_unet128_basic_64_content_gan_pixel_1')
# resnet_unet128_basic_64_A10_B10
# MODELS=('content_gan')

# MODEL='simple_pix2pix'
MODEL='gan_one_cycle'
# MODEL='content_gan'
PHASES=('eva')
# EPOCHS=(50 100 150 200)
EPOCHS=(200)
# EPOCH=latest
GPU=2
ASPECT_RATIO=1.0

# MODEL=

for EXPR_NAME in ${EXPR_NAMES[@]}; do
	for EPOCH in ${EPOCHS[@]}; do
		for PHASE in ${PHASES[@]}; do
			DATA_ROOT=../expr/$DB_NAME  name=${DB_NAME}_${EXPR_NAME} \
			checkpoints_dir=../checkpoints/ align_data=1 model=$MODEL which_direction='AtoB' aspect_ratio=$ASPECT_RATIO \
			results_dir=../results/${DB_NAME}/ \
			phase=$PHASE which_epoch=$EPOCH how_many='all' \
			gpu=$GPU nThreads=1 flip=0 display_id=$DISPLAY_ID serial_batches=1 \
			loadSize=128 fineSize=128 \
			th test.lua
		done
	done
done
