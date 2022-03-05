# Instructions for HW2 (Youngin Kim, youngin2)
HW2: 11-775 Large-Scale Multimedia Analysis, Spring 2022

## 0. Prerquisite

**Please modify `config.sh` for your own path**
- `BASE_DIR` : Base directory path that your *codes* are saved. 
- `P1_DATA_DIR` : **Part1** Data directory path that your data are saved.
- `P2_DATA_DIR` : **Part2** Data directory path that your data are saved.

Dependencies
```
conda env create -f environment.yml
conda activate 11775-hw2
```

## Part 1-1: SIFT + MLP

```
$ bash sh/sift_mlp.sh
```

## Part 1-2: CNN + MLP

```
$ bash sh/cnn_mlp.sh
```

## Part 1 - Validation Result
see `notebook/result.ipynb`

## Part 2
**Before run below code, you must need to define `BEST_MODEL_PATH` in `sh/part2.sh`**

`BEST_MODEL_PATH`: pretrained CNN model from part 1
```
$ bash sh/part2.sh
```
