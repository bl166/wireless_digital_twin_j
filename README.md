
# wireless digital twin (journal version)

This is an extension to our MILCOM paper:

Boning Li, Timofey Efimov, Abhishek Kumar, Jose Cortes, Gunjan Verma, Ananthram Swami, and Santiago Segarra. "[Learnable Digital Twin for Efficient Wireless Network Evaluation](https://arxiv.org/abs/2306.06574)." In *MILCOM*, IEEE (2023).

## organization

```
├── dataset/
│   ├── ns3-simulation-codes/
│   │   ├── data.ipynb                       # for data processing
│   │   ├── py_helpers.ipynb                 # for automating ns3 args
│   │   └── wireless-traffic-to-kpi5-ext.cc  # c++ code for ns3 simulation
│   ├── WifiGridFtR1.99S1St180Dr50Le2F4/     # datasets
│   └── ...
├── configs/
│   ├── wifigrid.ini
│   └── ...
├── utils/
│   ├── utilfunc.py
│   ├── ns3py.py
│   ├── ns3proc.py
│   ├── models.py       # implementation of digital twin models
│   ├── graph_convs.py  # customized gnn modules
│   └── datagen.py      # handling network data and KPIs label
├── train_univ_emb_batch.py   # main entrance for training
├── main-sow.ipynb            # for automating training args
└── main-harvest.ipynb        # for processing and visualizing results
```

## example

Run the training script
```
$ python train_univ_emb_batch.py -c ./configs/wifigrid.ini -m PlanNetEmbGCNBatch --loss mse -ep 100 -f 1
```

- `-c ./configs/wifigrid.ini` : training configurations, such as data/output paths, architecture, hyperparamters, training options, etc.
- `-m PlanNetEmbGCNBatch` : train a [PlanNetEmbGCNBatch](./utils/models.py#L484) model
- `--loss mse` : use MSE loss
- `-ep 100` : train for 100 epochs
- `-f 1` : cross-validation fold#1

## configs

Besides paths and architecture hyperparameters that are rather self-explanatory, we also have the following training options that may require further explanation to help understand them:

```
[LearningParams]
...
learn_embedding = True    # if False, fix parameters in embedding layers (for transfer learning)
label_scale     = True    # if False, use raw label values; if True, apply robust scaling
batching        = True    # if False, train one instance at a time (the batch_size set above will be ignored)
batch_norm      = False   # (not effective yet)
layer_norm      = False   # (not effective yet)
add_noise       = False   # if True, add gaussian noise to input (for random-flow experiments)
```

## model

We have implemented learnable digital twin architectures @[./utils/models.py](./utils/models.py) with the following features enabled:<br>
e.g. [PlanNetEmbGCNBatch](./utils/models.py#L484)

- **Multi-task learning**: shared embedding layers and individual readout layers, enabling joint learning of multiple KPIs and transfer learning using new KPI(s).
- **Parameter sharing**: the trainable parameters are identical across different layers for $t=1,...,T$, serving as a regularization against over-parameterization.
- **Batch training**:
- **Flexible node-embedding module**:



## TODOs

We are diligently working to finalize documentation for the following items:
- [ ] Data generation with ns3 and the following processing steps
- [ ] Data handling and loading for digital twin training
- [ ] Retraining process, or transfer learning
- [ ] Network optimization with trained digital twins
  - traffic loads management (gradient-based)
  - destination nodes offloading (hill-climbing)
