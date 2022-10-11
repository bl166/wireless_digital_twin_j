import tensorflow as tf
import numpy as np
import glob
import os,pdb
import datetime
from Datagen import get_data_generators, get_paths_from_routing
import configparser
from unoNet import unoNet
os.environ["CUDA_VISIBLE_DEVICES"]="-1"
if __name__=="__main__":
    
    config = configparser.ConfigParser()
    config.read('config.ini')

    paths = get_paths_from_routing(config['Paths']['routing'])
    traingen, valgen, testgen = get_data_generators(config['Paths']['data'],paths,config['Paths']['graph'])
    
    model = unoNet(config)
    model.build()
    model.compile()

    checkpoint_filepath = "setting-3/solo_gbn/logs/ckpt/20221010-123622"

    model.load_weights(checkpoint_filepath)

    delay_predicted = model.predict(testgen)
    delay_predicted = delay_predicted.reshape(testgen.__len__(),10) # 10 paths for this routing

    predictionSavePath = 'setting-3/solo_gbn/delay_gbn.txt'
    np.savetxt(predictionSavePath,delay_predicted,delimiter=",")

