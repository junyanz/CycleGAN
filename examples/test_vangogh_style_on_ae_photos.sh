#!/bin/sh

## This script download the dataset and pre-trained network,
## and generates style transferred images.

# Download the dataset. The downloaded dataset is stored in ./datasets/${DATASET_NAME}
DATASET_NAME='ae_photos'
bash ./datasets/download_dataset.sh $DATASET_NAME

# Download the pre-trained model. The downloaded model is stored in ./models/${MODEL_NAME}_pretrained/latest_net_G.t7
MODEL_NAME='style_vangogh'
bash ./pretrained_models/download_model.sh $MODEL_NAME

# Run style transfer using the downloaded dataset and model
DATA_ROOT=./datasets/$DATASET_NAME name=${MODEL_NAME}_pretrained model=one_direction_test phase=test how_many='all' loadSize=256 fineSize=256 resize_or_crop='scale_width' th test.lua

if [ $? == 0 ]; then
    echo "The result can be viewed at ./results/${MODEL_NAME}_pretrained/latest_test/index.html"
fi
