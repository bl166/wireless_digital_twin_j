

import tensorflow as tf
import numpy as np
import os
import datetime
from Datagen import get_data_generators, get_paths_from_routing
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
os.environ["CUDA_VISIBLE_DEVICES"]="1"


if __name__=="__main__":
    
    config = configparser.ConfigParser()
    config.read('config.ini')

    paths = get_paths_from_routing(config['Paths']['routing'])
    traingen, valgen, testgen = get_data_generators(config['Paths']['data'],paths,config['Paths']['graph'])
    
    initial_learning_rate = float(config['LearningParams']['learning_rate'])
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate,
        decay_steps=int(config['LearningParams']['lr_decay_steps']),
        decay_rate=float(config['LearningParams']['lr_decay_rate']),
        staircase=True)

    optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)

    log_dir = config['Paths']['logs']+"fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

    checkpoint_filepath = config['Paths']['logs']+"ckpt/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
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

    model.fit(traingen, 
        batch_size=int(config['LearningParams']['batch_size']), 
        epochs=int(config['LearningParams']['epochs']), 
        validation_data=valgen, 
        callbacks=[model_checkpoint_callback,tensorboard_callback]
        )




