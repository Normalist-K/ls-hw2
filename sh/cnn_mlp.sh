#!/bin/bash
source ./config.sh

cd $BASE_DIR
CSV=$P1_DATA_DIR/labels/total.csv

python code/run_cnn.py $CSV \
                       --video_dir $P1_DATA_DIR/videos \
                       --cnn_dir $P1_DATA_DIR/cnn_b4 \
                       --model_name efficientnet_b4 \
                       --img_size 380


python code/run_mlp.py cnn \
                       --feature_dir $P1_DATA_DIR/cnn_b4 \
                       --num_features 1792 \
                       --train_val_list_file $P1_DATA_DIR/labels/train_val.csv \
                       --test_list_file $P1_DATA_DIR/labels/test_for_students.csv \
                       --test_frac 0.2 \
                       --batch_size 512 \
                       --split_seed 42 \
                       --learning_rate 0.0001 \
                       --scheduler_T_max 500 \
                       --scheduler_eta_min 1e-6 \
                       --earlystop_patience 15 \