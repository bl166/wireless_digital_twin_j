import tensorflow as tf
import numpy as np
import os
import re
import datetime
import configparser

import utils.utilfunc as uf
from utils.models import PlanNet, RouteNet
import utils.datagen as dg


gpus = tf.config.list_physical_devices('GPU')
if gpus:
    # Restrict TensorFlow to only use the first GPU
    try:
        tf.config.set_visible_devices(gpus[0], 'GPU')
        logical_gpus = tf.config.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPU")
    except RuntimeError as e:
        # Visible devices must be set before GPUs have been initialized
        print(e)

print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
os.environ["CUDA_VISIBLE_DEVICES"]="1"


if __name__=="__main__":
    
    config  = uf.PathConfigParser('./configs/config.ini').as_dict()
    h_params = dict(**config['GNN'], **config['LearningParams'])

    datagens, datasets = dg.get_data_gens_sets(config)
        
    initial_learning_rate = float(h_params['learning_rate'])
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate,
        decay_steps = int(h_params['lr_decay_steps']),
        decay_rate  = float(h_params['lr_decay_rate']),
        staircase   = True)

    optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)

    model = PlanNet(h_params, train_on = config['Paths']['trainon'])
    # model = RouteNet(h_params, train_on = config['Paths']['trainon'])
    mclass = re.findall('\'.+\..+\.(.+)\'', str(model.__class__))[0] #<class 'utils.models.PlanNet'>

    # save model
    log_dir = os.path.join(config['Paths']['logs'][0], mclass, model.train_on)
    print(log_dir)
    os.makedirs(log_dir, exist_ok = True)
    
    tensorboard_callback = tf.keras.callbacks.TensorBoard(
        log_dir        = log_dir, 
        histogram_freq = 1
    )
    early_stop_callback = tf.keras.callbacks.EarlyStopping(
        monitor              = 'val_loss', 
        min_delta            = 0, 
        patience             = 10, 
        verbose              = 0, 
        mode                 = 'auto', 
        baseline             = None, 
        restore_best_weights = True
    )
    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=os.path.join(
            log_dir, "cp." + datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + ".ckpt"
        ),
        save_weights_only = True,
        monitor           = 'val_loss',
        mode              = 'min',
        save_freq         = 'epoch',
        save_best_only    = False,
        verbose           = 1
    )
    csv_callback = tf.keras.callbacks.CSVLogger(
        os.path.join(log_dir, 'training.log'), 
        separator = ',', 
        append    = False
    )

    model.build()
    model.compile(optimizer=optimizer)

    model.fit(datasets['train'], 
        batch_size      = int(h_params['batch_size']), 
        epochs          = int(h_params['epochs']), 
        validation_data = datasets['validate'], 
        callbacks = [
            model_checkpoint_callback,
            tensorboard_callback,
            csv_callback,
            early_stop_callback
        ]
    )


