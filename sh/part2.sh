#!/bin/bash
source ./config.sh

cd $BASE_DIR
CSV=$P2_DATA_DIR/labels_p2/test_for_students.csv
BEST_MODEL_PATH=$P1_DATA_DIR/mlp/cnn/version_2/checkpoints/epoch=30-step=371-val_acc=0.9440.ckpt
FEATURE_DIR=$P2_DATA_DIR/cnn_b4

python code/run_cnn.py $CSV \
                       --video_dir $P2_DATA_DIR/videos_p2 \
                       --cnn_dir $FEATURE_DIR \

python code/infer_mlp.py --best_model_path $BEST_MODEL_PATH \
                         --list_file_path $CSV \
                         --feature_dir $FEATURE_DIR \
                         --save_csv_path $P2_DATA_DIR/infer/eff_b4_test_prediction.csv