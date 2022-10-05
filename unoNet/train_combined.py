

import tensorflow as tf
import numpy as np
import pandas as pd
import spektral as spk
import glob
import os,pdb
import datetime
from Datagen import get_data_generators, combinedDataGens
import configparser
from unoNet import unoNet


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
os.environ["CUDA_VISIBLE_DEVICES"]="0"


if __name__=="__main__":
    
    config = configparser.ConfigParser()
    config.read('config.ini')

    # NSFNet Routing-1
    paths = np.array([[0,1,7], [1,7,10,11], [2,5,13], [3,0], [6,4,5], [9,12], [10,9,8], [11,10], [12,9], [13,5,4,3]])
    dataPath = '/root/wireless_dataset_v0/ns-3.35/WirelessDataset/digi_twin_summer_wireless_dataset/NSFNet_routing_1'
    traingen, valgen, testgen = get_data_generators(dataPath,paths,config)
    
    # NSFNet Routing-2
    paths2 = np.array([[9, 10, 7, 1],[3, 4, 5, 13],[12, 5, 2, 0],[10, 9, 8, 3],
                      [13, 10, 7],[1, 0, 3, 8],[2, 5, 12, 9],[11, 12, 5, 2],
                      [6, 4, 5, 12],[5, 12, 11],[0, 1, 7, 10],[7, 6, 4]])
    dataPath2 = '/root/wireless_dataset_v0/ns-3.35/WirelessDataset/digi_twin_summer_wireless_dataset/NSFNet_routing_2'
    traingen2, valgen2, testgen2 = get_data_generators(dataPath2,paths2,config)
    
    # pdb.set_trace()
    # Combining datasets
    traingen_comb = combinedDataGens(traingen,traingen2)
    valgen_comb = combinedDataGens(valgen,valgen2)
    
    initial_learning_rate = float(config['LearningParams']['learning_rate'])

    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate,
        decay_steps=int(config['LearningParams']['lr_decay_steps']),
        decay_rate=float(config['LearningParams']['lr_decay_rate']),
        staircase=True)

    optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)

    log_dir = "logsComb/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

    checkpoint_filepath = "logsComb/ckpt/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_filepath,
        save_weights_only=True,
        monitor='val_accuracy',
        mode='max',
        save_freq='epoch',
        save_best_only=False)

    model = unoNet(config)
    model.build()
    model.compile(optimizer=optimizer)

    model.fit(traingen_comb, 
        batch_size=int(config['LearningParams']['batch_size']), 
        epochs=int(config['LearningParams']['epochs']), 
        validation_data=valgen_comb, 
        callbacks=[model_checkpoint_callback,tensorboard_callback]
        )