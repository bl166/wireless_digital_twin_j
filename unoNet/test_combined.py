import tensorflow as tf
import numpy as np
import glob
import os,pdb
import datetime
from Datagen import get_data_generators,combinedDataGens, get_paths_from_routing
import configparser
from unoNet import unoNet
os.environ["CUDA_VISIBLE_DEVICES"]="-1"
if __name__=="__main__":
    
    config = configparser.ConfigParser()
    config.read('config_combined.ini')

    # For combined training
    routingList = config['Paths']['routing'].split()
    dataList = config['Paths']['data'].split()
    graphList = config['Paths']['graph'].split()
  
    trainGenList = []
    testGenList = []
    valGenList= []
    
    for i in range(len(routingList)):
        paths = get_paths_from_routing(routingList[i])
        trainGen, valGen, testGen = get_data_generators(dataList[i],paths,graphList[i])
        trainGenList.append(trainGen)
        valGenList.append(valGen)
        testGenList.append(testGen)
    
    model = unoNet(config)
    model.build()
    model.compile()

    checkpoint_filepath = "setting-3/joint/logs/ckpt/20221011-064848"

    model.load_weights(checkpoint_filepath)
    model.evaluate(testGenList[0]) # Evaluating on first dataset
    model.evaluate(testGenList[1]) # Evaluating on second dataset
    
    # savePath = "setting-3/joint/"

    # pdb.set_trace()
    # savePath1 = savePath+"delay_nsfnet_routing1.txt"
    # delay_predicted = model.predict(testGenList[0])
    # delay_predicted = delay_predicted.reshape(testGenList[0].__len__(),10)
    # np.savetxt(savePath1,delay_predicted,delimiter=",")
    
    # savePath2 = savePath+"delay_gbn.txt"
    # delay_predicted = model.predict(testGenList[1])
    # delay_predicted = delay_predicted.reshape(testGenList[1].__len__(),10)
    # np.savetxt(savePath2,delay_predicted,delimiter=",")
