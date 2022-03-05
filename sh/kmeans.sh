#!/bin/bash
source ./config.sh

cd $BASE_DIR
CSV=$P1_DATA_DIR/labels/total.csv


python code/train_kmeans.py $CSV \
                            $P1_DATA_DIR/sift \
                            128 \
                            sift_128 \
                            --model_dir $P1_DATA_DIR/kmeans \
                            --seed 42 \
                            --debug


# python code/train_kmeans.py $P2_DATA_DIR/labels_p2/debug.csv \
#                             $P2_DATA_DIR/sift \
#                             12 \
#                             sift_128 \
#                             --model_dir $P2_DATA_DIR/kmeans \
#                             --seed 42 \
#                             --debug

# python code/train_kmeans.py $P2_DATA_DIR/labels_p2/test_for_students.csv \
#                             $P2_DATA_DIR/sift \
#                             128 \
#                             sift_128 \
#                             --model_dir $P2_DATA_DIR/kmeans \
#                             --seed 42 \
