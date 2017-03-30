FILE=$1
URL=https://people.eecs.berkeley.edu/~taesung_park/projects/CycleGAN/models/$FILE.t7
MODEL_FILE=./models/$FILE.t7
wget -N $URL -O $MODEL_FILE
