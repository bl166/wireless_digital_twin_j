
# wireless_digital_twin

Download the latest dataset `20230405.zip` from: https://drive.google.com/drive/folders/1fNyC9nHHppqyhet2WCWrTbdTVCh63T9U?usp=sharing

- *Note: You may need to change the paths in `gfiles.txt` to the `graph/` locations on your machine.*

## Example

Run the training script

```
CUDA_VISIBLE_DEVICES=0 python train.py -c ./configs/nsf-delay.ini -m p -f 1
```

- `CUDA_VISIBLE_DEVICES=0` ==> Use GPU#0 for training.
- `-c ./configs/nsf-delay.ini` ==> Training configurations, like data/output paths, architecture, hyperparamters, etc. 
- `-m p` ==> The model is PLAN-Net. Replace it with `-m r` to train a RouteNet
- `-f 1` ==> Cross-validation fold#1. Also legit: `-f 2` and `-f 3`.


## PLAN-Net Algorithm
![alt text](./figs/plannet-algo.png?raw=true)
