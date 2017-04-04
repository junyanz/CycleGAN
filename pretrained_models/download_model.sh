FILE=$1

echo "Note: available models are apple2orange, facades_photo2label, map2sat, orange2apple, style_cezanne, style_ukiyoe,  summer2winter_yosemite, zebra2horse, facades_label2photo, horse2zebra,monet2photo, sat2map, style_monet,style_vangogh, winter2summer_yosemite, iphone2dslr_flower"

echo "Specified [$FILE]"

mkdir -p ./checkpoints/${FILE}_pretrained
URL=https://people.eecs.berkeley.edu/~taesung_park/CycleGAN/models/$FILE.t7
MODEL_FILE=./checkpoints/${FILE}_pretrained/latest_net_G.t7
wget -N $URL -O $MODEL_FILE
