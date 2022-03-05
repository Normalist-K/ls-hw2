#!/bin/bash
source ./config.sh

cd $BASE_DIR
CSV=$P1_DATA_DIR/labels/total.csv

python code/run_sift.py $CSV \
                        --video_dir $P1_DATA_DIR/videos \
                        --sift_dir $P1_DATA_DIR/subset/sift \

python code/train_kmeans.py $CSV \
                            $P1_DATA_DIR/sift \
                            128 \
                            sift_128 \
                            --model_dir $P1_DATA_DIR/kmeans \
                            --seed 42 \

python code/run_bow.py $CSV \
                       sift_128 \
                       $P1_DATA_DIR/sift \
                       --model_dir $P1_DATA_DIR/kmeans \
                       --bow_dir_prefix $P1_DATA_DIR/bow \

python code/run_mlp.py sift \
                       --feature_dir $P1_DATA_DIR/bow_sift_128 \
                       --num_features 128 \
                       --train_val_list_file $P1_DATA_DIR/labels/train_val.csv \
                       --test_list_file $P1_DATA_DIR/labels/test_for_students.csv \
                       --test_frac 0.2 \
                       --batch_size 512 \
                       --split_seed 42 \
                       --learning_rate 0.0001 \
                       --scheduler_T_max 500 \
                       --scheduler_eta_min 1e-6 \
                       --earlystop_patience 15 \