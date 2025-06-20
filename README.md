# WaveRec
The source code for WaveRec

## Quick Start
## Example 1
python main.py --model_type waverec --data_name Beauty --lr 0.0005 --filter_type db2 --pass_weight=0.7 --train_name WaveRec_Beauty

## Example 2
python main.py --model_type waverec --data_name ML-1M --lr 0.0005 --filter_type meyer --pass_weight 0.7 --filter_length 8 --sigma 1.0 --train_name WaveRec_ML-1M

## Acknowledgement
This repository is based on [BSARec](https://github.com/yehjin-shin/BSARec).
