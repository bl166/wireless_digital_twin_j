import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
import os
import re
import datetime
import configparser
import argparse
import shutil

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
    
    # Instantiate the parser
    parser = argparse.ArgumentParser()
    parser.add_argument('-c','--config', type=str, default='./configs/config.ini')
    parser.add_argument('-m','--model', type=str, default='plannet')
    parser.add_argument('-f','--cvfold', type=int, default=0)
    parser.add_argument('--log_conversion', action='store_true')
    parser.add_argument('-r', '--resume', action='store_true')
    parser.add_argument('--no-resume', dest='resume', action='store_false')
    parser.set_defaults(resume=True)
    args = parser.parse_args()
    
    ##
    cfg_file = args.config #'./configs/config.ini'
    print(cfg_file)
    config  = uf.PathConfigParser(cfg_file).as_dict()
    if args.cvfold != 0:
        config['Paths']['logs'][0] += f'/cv{args.cvfold}/'
    h_params = dict(**config['GNN'], **config['LearningParams'])

    datagens, datasets = dg.get_data_gens_sets(config, fold=args.cvfold)
        
    initial_learning_rate = float(h_params['learning_rate'])
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate,
        decay_steps = int(h_params['lr_decay_steps']),
        decay_rate  = float(h_params['lr_decay_rate']),
        staircase   = True)

    optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)

    mcode = args.model.lower()[0]
    if mcode == 'p':
        model = PlanNet(h_params, log_conversion=args.log_conversion, train_on = config['Paths']['trainon'])
    elif mcode == 'r':     
        model = RouteNet(h_params, log_conversion=args.log_conversion , train_on = config['Paths']['trainon'])
    mclass = re.findall('\'.+\..+\.(.+)\'', str(model.__class__))[0] #<class 'utils.models.PlanNet'>

    # save model
    model_dirname = 'TRG+' + model.train_on + '_DS+' + '+'.join(
        #[' '.join(re.split('_|-',os.path.basename(d))).title().replace(' ', '') for d in config['Paths']['data']]
        [os.path.basename(d) for d in config['Paths']['data']]
    )
    log_dir = os.path.join(config['Paths']['logs'][0], mclass, model_dirname+'_log5')
    os.makedirs(log_dir, exist_ok = True)
    shutil.copyfile(cfg_file, os.path.join(log_dir, os.path.basename(cfg_file)))
    
    iters_per_epoch = datagens['train'].__len__()
    tensorboard_callback = tf.keras.callbacks.TensorBoard(
        log_dir        = log_dir, 
        histogram_freq = 1
    )
    early_stop_callback = tf.keras.callbacks.EarlyStopping(
        monitor              = 'val_loss', 
        min_delta            = 0, 
        patience             = 20, 
        verbose              = 0, 
        mode                 = 'auto', 
        baseline             = None, 
        restore_best_weights = True
    )
    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        os.path.join(log_dir, "cp-{epoch:04d}.ckpt"),
        save_freq         = iters_per_epoch*5,
        save_best_only    = False,
        save_weights_only = True,
        verbose           = 1
    )
    model_best_loss_callback = tf.keras.callbacks.ModelCheckpoint(
        os.path.join(log_dir, "cp-best-loss.ckpt"),
        monitor           = 'val_loss',
        mode              = 'min',
        save_best_only    = True,
        save_weights_only = True,
        verbose           = 1
    )
    model_best_mae_callback = tf.keras.callbacks.ModelCheckpoint(
        os.path.join(log_dir, "cp-best-mae.ckpt"),
        monitor           = 'val_mae',
        mode              = 'min',
        save_best_only    = True,
        save_weights_only = True,
        verbose           = 1
    )
    model_latest_callback = tf.keras.callbacks.ModelCheckpoint(
        os.path.join(log_dir, "cp-latest.ckpt"),
        save_freq         = 'epoch',
        save_best_only    = False,
        save_weights_only = False,
        verbose           = 0
    )
    csv_callback = tf.keras.callbacks.CSVLogger(
        os.path.join(log_dir, 'training.log'), 
        separator = ',', 
        append    = True
    )

    # resume training if possible
    model.build()
    model.compile(optimizer=optimizer, run_eagerly=False)
    initial_epoch = 0
    if args.resume and os.path.exists(model_latest_callback.filepath):
        print('Loading weights from', model_latest_callback.filepath)
        ckpt = tf.train.Checkpoint(model)
        ckpt.restore(model_latest_callback.filepath)
        #model.load_weights(model_latest_callback.filepath)
        initial_epoch = model.optimizer.iterations.numpy() // iters_per_epoch

    model.fit(
        x = datasets['train'].shuffle(datagens['train'].n, reshuffle_each_iteration=True),
        batch_size      = int(h_params['batch_size']), 
        epochs          = int(h_params['epochs']), 
        initial_epoch   = initial_epoch,
        validation_data = datasets['validate'], 
        callbacks       = [
            model_checkpoint_callback,
            model_best_loss_callback,
            model_best_mae_callback,
            model_latest_callback,
            tensorboard_callback,
            csv_callback,
            early_stop_callback
        ]
    )


