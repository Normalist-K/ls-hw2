#!/bin/bash
source ./config.sh

cd $BASE_DIR

python code/run_mlp.py cnn \
                       --feature_dir $P1_DATA_DIR/cnn_resnet50 \
                       --num_features 2048 \
                       --train_val_list_file $P1_DATA_DIR/labels/train_val.csv \
                       --test_list_file $P1_DATA_DIR/labels/test_for_students.csv \
                       --test_frac 0.2 \
                       --batch_size 512 \
                       --split_seed 42 \
                       --learning_rate 0.0001 \
                       --scheduler_T_max 500 \
                       --scheduler_eta_min 1e-6 \
                       --earlystop_patience 15 \

# python code/run_mlp.py cnn \
#                        --feature_dir $P1_DATA_DIR/cnn_b4 \
#                        --num_features 1792 \
#                        --train_val_list_file $P1_DATA_DIR/labels/train_val.csv \
#                        --test_list_file $P1_DATA_DIR/labels/test_for_students.csv \
#                        --test_frac 0.2 \
#                        --batch_size 512 \
#                        --split_seed 42 \
#                        --learning_rate 0.0001 \
#                        --scheduler_T_max 500 \
#                        --scheduler_eta_min 1e-6 \
#                        --earlystop_patience 15 \

# python code/run_mlp.py cnn \
#                        --feature_dir $P1_DATA_DIR/cnn \
#                        --num_features 1280 \
#                        --train_val_list_file $P1_DATA_DIR/labels/train_val.csv \
#                        --test_list_file $P1_DATA_DIR/labels/test_for_students.csv \
#                        --test_frac 0.2 \
#                        --batch_size 512 \
#                        --split_seed 42 \
#                        --learning_rate 0.0001 \
#                        --scheduler_T_max 500 \
#                        --scheduler_eta_min 1e-6 \
#                        --earlystop_patience 15 \

# python code/run_mlp.py sift \
#                        --feature_dir $P1_DATA_DIR/bow_sift_128 \
#                        --num_features 128 \
#                        --train_val_list_file $P1_DATA_DIR/labels/train_val.csv \
#                        --test_list_file $P1_DATA_DIR/labels/test_for_students.csv \
#                        --test_frac 0.2 \
#                        --batch_size 512 \
#                        --split_seed 42 \
#                        --learning_rate 0.0001 \
#                        --scheduler_T_max 500 \
#                        --scheduler_eta_min 1e-6 \
#                        --earlystop_patience 15 \