#!/bin/bash
source ./config.sh

cd $BASE_DIR

# ipython --pdb code/run_sift.py $P2_DATA_DIR/labels_p2/debug.csv \
#                         --video_dir $P2_DATA_DIR/videos_p2 \
#                         --sift_dir $P2_DATA_DIR/sift \
#                         --debug

python code/run_sift.py $P2_DATA_DIR/labels_p2/test_for_students.csv \
                        --video_dir $P2_DATA_DIR/videos_p2 \
                        --sift_dir $P2_DATA_DIR/sift \


# python code/run_sift.py /root/class/cmu/LSMA/ls-hw2/video_not_found.csv \
#                         --video_dir $P2_DATA_DIR/videos_p2 \
#                         --sift_dir $P2_DATA_DIR/sift \
#                         --debug