#!/bin/bash
source ./config.sh

cd $BASE_DIR

# python code/run_bow.py $P2_DATA_DIR/labels_p2/debug.csv \
#                        sift_128 \
#                        $P2_DATA_DIR/sift \
#                        --model_dir $P2_DATA_DIR/kmeans \
#                        --bow_dir_prefix $P2_DATA_DIR/bow \
#                        --debug

python code/run_bow.py $P2_DATA_DIR/labels_p2/test_for_students.csv \
                       sift_128 \
                       $P2_DATA_DIR/sift \
                       --model_dir $P2_DATA_DIR/kmeans \
                       --bow_dir_prefix $P2_DATA_DIR/bow \