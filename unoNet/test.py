import tensorflow as tf
import numpy as np
import glob
import os,pdb
import datetime
from Datagen import get_data_generators
import configparser
from unoNet import unoNet

if __name__=="__main__":
    
    config = configparser.ConfigParser()
    config.read('config.ini')

    # NSFNet Routing-1
    paths = np.array([[0,1,7], [1,7,10,11], [2,5,13], [3,0], [6,4,5], [9,12], [10,9,8], [11,10], [12,9], [13,5,4,3]])
    dataPath = '/root/wireless_dataset_v0/ns-3.35/WirelessDataset/digi_twin_summer_wireless_dataset/NSFNet_routing_1'
    traingen, valgen, testgen,test2gen = get_data_generators(dataPath,paths,config)
    
    model = unoNet(config)
    model.build()
    model.compile()

    checkpoint_filepath = "logs/ckpt/20221004-120050"

    model.load_weights(checkpoint_filepath)

    model.evaluate(testgen)

    delay_predicted = model.predict(testgen)
    delay_predicted = delay_predicted.reshape(1000,10)

    predictionSavePath = '/root/wireless_dataset_v0/tf_codes/delay_unoNet.txt'
    np.savetxt(predictionSavePath,delay_predicted,delimiter=",")

