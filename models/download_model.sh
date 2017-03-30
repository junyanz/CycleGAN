FILE=$1

echo "Note: available models are apple2orange, facades_photo2label, map2sat, orange2apple, style_cezanne, style_ukiyoe,  summer2winter_yosemite, zebra2horse, facades_label2photo, horse2zebra,monet2photo, sat2map, style_monet,style_vangogh, winter2summer_yosemite"

echo "Specified [$FILE]"

URL=https://people.eecs.berkeley.edu/~taesung_park/CycleGAN/models/$FILE.t7
wget $URL -O ./models/$FILE.t7
