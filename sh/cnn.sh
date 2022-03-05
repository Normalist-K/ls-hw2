#!/bin/bash
source ./config.sh

cd $BASE_DIR
CSV=$P1_DATA_DIR/labels/total.csv

# python code/run_cnn.py $P2_DATA_DIR/labels_p2/debug.csv \
#                        --video_dir $P2_DATA_DIR/videos_p2 \
#                        --cnn_dir $P2_DATA_DIR/cnn \
#                        --debug \

# python code/run_cnn.py $P2_DATA_DIR/labels_p2/test_for_students.csv \
#                        --video_dir $P2_DATA_DIR/videos_p2 \
#                        --cnn_dir $P2_DATA_DIR/cnn_b4 \

python code/run_cnn.py $CSV \
                       --video_dir $P1_DATA_DIR/videos \
                       --cnn_dir $P1_DATA_DIR/cnn_b0_ \
                       --model_name resnet50 \
                       --img_size 224