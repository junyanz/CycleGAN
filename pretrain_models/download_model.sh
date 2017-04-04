FILE=$1
mkdir -p ./checkpoints/${FILE}_pretrained
URL=https://people.eecs.berkeley.edu/~taesung_park/CycleGAN/models/$FILE.t7
MODEL_FILE=./checkpoints/${FILE}_pretrained/latest_net_G.t7
wget -N $URL -O $MODEL_FILE
