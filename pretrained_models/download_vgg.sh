URL1=https://people.eecs.berkeley.edu/~taesung_park/projects/CycleGAN/models/places_vgg.caffemodel
MODEL_FILE1=./models/places_vgg.caffemodel
URL2=https://people.eecs.berkeley.edu/~taesung_park/projects/CycleGAN/models/places_vgg.prototxt
MODEL_FILE2=./models/places_vgg.prototxt
wget -N $URL1 -O $MODEL_FILE1
wget -N $URL2 -O $MODEL_FILE2
