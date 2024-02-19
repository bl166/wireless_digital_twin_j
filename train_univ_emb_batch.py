import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
import pandas as pd
import os
import re
import datetime
import configparser
import argparse
import shutil
import glob
from tqdm.auto import tqdm

import utils.utilfunc as uf
from utils.models import PlanNetEmb as PlanNetEmbECCBatch
from utils.models import PlanNetEmbGCN as PlanNetEmbGCNBatch
from utils.models import RouteNetEmb
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
    parser.add_argument('--loss', type=str, default='mse')
    parser.add_argument('--optm', type=str, default='adam')
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--suffix', type=str, default='')

    parser.add_argument('--freeze', action='store_true')
    parser.add_argument('-lw','--loadweights', type=str, default='')
    parser.add_argument('-es','--early_stop', type=int, default=0)
    parser.add_argument('-ep','--epochs', type=int, default=0)

    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()

    ##
    cfg_file = args.config
    print(cfg_file)
    config  = uf.PathConfigParser(cfg_file).as_dict()
    if args.cvfold != 0:
        config['Paths']['logs'][0] += f'/cv{args.cvfold}/'
    h_params = dict(**config['GNN'], **config['LearningParams'])

    ##
    n_epochs = int(h_params['epochs']) if not args.epochs else args.epochs

    ##
    print(config)
    datagens, datasets = dg.get_data_gens_sets(config, fold=args.cvfold)

    ##
    initial_learning_rate = float(h_params['learning_rate'])
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate,
        decay_steps = int(h_params['lr_decay_steps']),
        decay_rate  = float(h_params['lr_decay_rate']),
        staircase   = True)

    ##
    if args.optm.lower()=='adam':
        optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
    elif args.optm.lower()=='sgd':
        optimizer = tf.keras.optimizers.SGD(learning_rate=initial_learning_rate)
    else:
        raise

    m_argkw = {
        'final_activation': config['Paths']['fin_activ'][0] if 'fin_activ' in config['Paths'] else None,
        'train_on': config['Paths']['trainon'],
        'loss': args.loss,
        'sharing': h_params['sharing'] if 'sharing' in h_params else True,
        'add_noise': h_params['add_noise'] if 'add_noise' in h_params else False,
    }

    if args.freeze:
        h_params['learn_embedding'] = False

    ModelClass = globals()[args.model]
    #batching = 'batch' in args.model.lower()
    batching = h_params['batching'] and (h_params['batch_size']>0)
    if batching:
        m_argkw['batching'] = True
        m_argkw['batch_norm'] = h_params['batch_norm'] and (h_params['batch_size']>1)

    model = ModelClass(h_params, **m_argkw)
    model.build()

    # transfer learning
    if args.freeze:
        model.path_update.trainable = False
        model.edge_update.trainable = False
        model.node_update.trainable = False

    model.compile(
        optimizer   = optimizer,
        loss = m_argkw['loss'],
        run_eagerly = args.debug
    )
    #print(uf.get_layers_weights(model)['path_update'])

    print('len(model.trainable_weights) =', len(model.trainable_weights))
    if args.loadweights:
        print()
        load_ini_file = glob.glob(os.path.join(os.path.dirname(args.loadweights), '*.ini'))[0]
        m_argkw['train_on'] = uf.PathConfigParser(load_ini_file).as_dict()['Paths']['trainon']
        load_model = ModelClass(h_params, **m_argkw)
        load_model.build()
        load_model.compile(optimizer, loss=m_argkw['loss'])
        #load_model(datagens['train'].__getitem__(0)[0])

        print('Loading pretrained weights from', args.loadweights)
        ckpt = tf.train.Checkpoint(load_model)
        ckpt.restore(args.loadweights).expect_partial()

        model.path_update.set_weights(load_model.path_update.get_weights())
        model.edge_update.set_weights(load_model.edge_update.get_weights())
        model.node_update.set_weights(load_model.node_update.get_weights())
        for lab in model.train_on:
            if lab in list(load_model.readout):
                print(lab, ': loading readout layers...')
                model.readout[lab].set_weights(load_model.readout[lab].get_weights())
            if lab in list(load_model.final):
                print(lab, ': loading final layer...')
                model.final[lab].set_weights(load_model.final[lab].get_weights())

    mclass = re.findall('\'.+\..+\.(.+)\'', str(model.__class__))[0] #<class 'utils.models.PlanNetEmb'>
    print()

    # save model
    model_dirname = 'TRG+' + '+'.join(model.train_on) + '_DS+' + '+'.join(
        [os.path.basename(d) for d in config['Paths']['data']]
    )
    print(model_dirname)

    log_dir = os.path.join(
        config['Paths']['logs'][0],
        args.model,
        model_dirname+f'_LR{initial_learning_rate:.2e}'
    )
    if batching:
        log_dir += '_BS'+str(h_params['batch_size'])
    if not m_argkw['sharing']:  #whether to share parameters accross layers
        log_dir += '_noShare'

    log_dir += '_LF+'+args.loss #loss function

    if 'add_noise' in m_argkw and m_argkw['add_noise']:
        log_dir += '_wNoise'

    log_dir += f'_{args.suffix}'
    os.makedirs(log_dir, exist_ok = True)
    try:
        shutil.copyfile(
            cfg_file,
            os.path.join(log_dir, os.path.basename(cfg_file))
        )
    except shutil.SameFileError:
        pass

    # callback and checkpoint settings
    total_tr, total_va = datagens['train'].n, datagens['validate'].n
    iters_per_epoch   = total_tr if not batching else -(-total_tr//model.batch_size)
    cb_csvlogger_path = os.path.join(log_dir, 'training.log')
    cb_ckpt_at_path   = lambda x: os.path.join(log_dir, f'cp-{x:04d}.ckpt')
    cb_latest_path    = os.path.join(log_dir, "cp-latest.ckpt")
    cb_best_paths     = lambda x: os.path.join(log_dir, f'cp-best-{x}.ckpt')


    """
    CUDA_VISIBLE_DEVICES=-1 python train_univ.py --suffix x_debug_retrain_2 -c ./configs/nsf-delay-jitter-throughput-drops-dr100.ini -m PlanNet -f 1 --loadweights ./output_j/cv1/PlanNet/TRG+delay+jitter_DS+NsfnetR1S1St180Dr100_LR5.00e-04_x_debug_/cp-0100.ckpt
    """
    # resume training if possible
    initial_epoch = 0
    if args.resume and (os.path.exists(cb_latest_path) or os.path.exists(cb_latest_path+'.index')):
        print('Loading weights from', cb_latest_path)
        ckpt = tf.train.Checkpoint(model)
        ckpt.restore(cb_latest_path).expect_partial()
        initial_epoch = model.optimizer.iterations.numpy() // iters_per_epoch
        print('Resuming from epoch', initial_epoch, model.optimizer.iterations.numpy(), iters_per_epoch )

    print('len(model.trainable_weights) =', len(model.trainable_weights))
    for block in model.layers:
        print(block.name, block.trainable)
        try:
            for layer in block.layers:
                print('\t', layer.name, layer.trainable)
        except:
            pass

    print()
    train_ds = datasets['train'].shuffle(total_tr//3, reshuffle_each_iteration=True)
    valid_ds = datasets['validate']#.take(model.batch_size)
    if batching:
        train_ds = dg.dataset_batchify(train_ds, model.batch_size, padding=True)
        valid_ds = dg.dataset_batchify(valid_ds, model.batch_size, padding=True)
    else:
        valid_ds = valid_ds.take(1)


    ################################
    #    MY TRAINING FRAMEWORK     #
    ################################
    try:
        monitors = {'loss': float('inf'), 'mae': float('inf')}
        learning_log = []
        escount = 0
        for ep in range(initial_epoch, n_epochs):
            if args.early_stop > 0 and escount > args.early_stop:
                raise KeyboardInterrupt

            train_log = {}
            pbar = tqdm( train_ds, total=iters_per_epoch)
            for step, batch in enumerate(pbar):
                log = model.train_step(batch)

                if not train_log:
                    train_log = log
                else:
                    for k in log.keys():
                        train_log[k] += log[k]

                pbar.set_description(
                    f"Train: epoch {ep:3d}, step {step:4d}, Loss: {train_log['loss']/(step+1):8f}"
                )

            # average for the current epoch
            train_log = {
                **{'epoch': ep},
                **{k:v.numpy() / (step+1) for k,v in train_log.items()}
            }

            # checkpoints
            model.save_weights(cb_ckpt_at_path(ep+1))

            # validation
            valid_log = {}
            pbar = tqdm( valid_ds , total=total_va if not batching else -(-total_va//model.batch_size))
            for step, batch in enumerate(pbar):
                log = model.test_step(batch)

                if not valid_log:
                    valid_log = log
                else:
                    for k in log.keys():
                        valid_log[k] += log[k]
                pbar.set_description(
                    f"Valid: epoch {ep:3d}, step {step:4d}, Loss: {valid_log['loss']/(step+1):8f}"
                )
            # average
            valid_log = {'val_'+k : v.numpy() / (step+1) for k,v in valid_log.items()}
            print("Validation MAE Loss is: {}".format(valid_log['val_mae']))

            # merge
            learning_log.append({**train_log, **valid_log})

            # save to csv
            df_csv = pd.DataFrame([learning_log[-1]])
            df_csv.to_csv(
                cb_csvlogger_path,
                index=False,
                mode='a',
                header=not os.path.exists(cb_csvlogger_path)
            ) # convert DataFrame to csv

            # best model and early stop
            esincrement = 0
            for mi, m in enumerate(['loss', 'mae']):
                metr_new = learning_log[-1][f'val_{m}']
                if metr_new <= monitors[m]:
                    best_path = cb_best_paths(m)
                    print('val_{} improved from {} to {}, saving to {}'.format(
                        m, monitors[m], metr_new, best_path)
                    )
                    model.save_weights(best_path)
                    monitors[m] = metr_new
                else:
                    esincrement += 1
            escount += (-escount) ** (esincrement<mi+1) #increment or reset

            model.save_weights(cb_latest_path)

    except KeyboardInterrupt:
        print('exit and saved to', cb_latest_path)
        model.save_weights(cb_latest_path)

    print('Done -->', log_dir)
