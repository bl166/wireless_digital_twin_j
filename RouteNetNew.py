import os

##DataLoc

dat0 = '/root/data/'
dat1 = '/*.tfrecords'

import glob

#Train
loc = 'train'

full_train = glob.glob(dat0+loc+dat1)
print('len of train is ')
print(len(full_train))

#Test 1
loc = 'test-1'

full_test1 = glob.glob(dat0+loc+dat1)
print('Length of full test1 is')
print(len(full_test1))

#Test 2
loc = 'test-2'

full_test2 = glob.glob(dat0+loc+dat1)

# import random
# def create_validation_set(full_train, split_ratio):
#     """
#     Inputs:
#         -full_train: the list of paths to the files that are for training
#         -split_ratio: a float between 0 and 1 to which portion of files from training
#                         to transfer to validation dataset
#     """
#     full_valid = []
#     size_of_training = len(full_train)
#     for i in range(int(split_ratio*size_of_training)):
#         file_to_validation = random.choice(full_train)
#         full_valid.append(file_to_validation)
#         full_train.remove(file_to_validation)
#     return full_train, full_valid
# print("After split")
# full_train_valid = create_validation_set(full_train, 0.1)
# full_train = full_train_valid[0]
# full_valid = full_train_valid[1]
# print(len(full_train))
# print(len(full_valid))

#validation
# import random
# def create_validation_set(full_train, split_ratio):
#     """
#     Inputs:
#         -full_train: the list of paths to the files that are for training
#         -split_ratio: a float between 0 and 1 to which portion of files from training
#                         to transfer to validation dataset
#     """
#     full_valid = []
#     size_of_training = len(full_train)
#     for i in range(int(split_ratio*size_of_training)):
#         file_to_validation = random.choice(full_train)
#         full_valid.append(file_to_validation)
#         full_train.remove(file_to_validation)
#     return full_train, full_valid
# print("After split")
# full_train_valid = create_validation_set(full_train, 0.12)
# full_train = full_train_valid[0]
# full_valid = full_train_valid[1]
# print(len(full_train))
# print(len(full_valid))

#Validation
loc = 'validate'

full_valid = glob.glob(dat0+loc+dat1)
print('len of validation set is')
print(len(full_valid))

##TF Code
from tensorflow.python import debug as tf_debug
from heapq import merge
import tensorflow as tf
from tensorflow import keras
import numpy as np
import argparse
import pdb 

##Model

class RouteNet(tf.keras.Model):
    def __init__(self,hparams, output_units=1, final_activation=None):
        super(RouteNet, self).__init__()

        self.hparams = hparams
        self.output_units = output_units
        self.final_activation = final_activation

        
    def build(self, input_shape=None):
        del input_shape
        
        self.edge_update = tf.keras.layers.GRUCell(self.hparams.link_state_dim, name="edge_update")
        self.path_update = tf.keras.layers.GRUCell(self.hparams.path_state_dim, name="path_update")

        # self.relu = tf.keras.layers.ReLU(self.hparams.path_state_dim, name='attention')
        # self.relu1 = tf.keras.layers.ReLU(self.hparams.path_state_dim, name='attention1')
        self.relu2 = tf.keras.layers.ReLU(self.hparams.path_state_dim, name='attention2')


        self.readout = tf.keras.models.Sequential(name='readout')

        for i in range(self.hparams.readout_layers):
            self.readout.add(tf.keras.layers.Dense(self.hparams.readout_units, 
                    activation=tf.nn.selu,
                    kernel_regularizer=tf.contrib.layers.l2_regularizer(self.hparams.l2)))
            self.readout.add(tf.keras.layers.Dropout(rate=self.hparams.dropout_rate))

        self.final = keras.layers.Dense(self.output_units, 
                kernel_regularizer=tf.contrib.layers.l2_regularizer(self.hparams.l2_2),
                activation = self.final_activation )
        
        self.edge_update.build(tf.TensorShape([None,self.hparams.path_state_dim]))
        self.path_update.build(tf.TensorShape([None,self.hparams.link_state_dim]))
        
        # self.relu.build(tf.TensorShape([None,self.hparams.path_state_dim]))
        # self.relu1.build(tf.TensorShape([None,self.hparams.path_state_dim]))
        self.relu2.build(tf.TensorShape([None,self.hparams.path_state_dim]))
        
        self.readout.build(input_shape = [None,self.hparams.path_state_dim])
        self.final.build(input_shape = [None,self.hparams.path_state_dim + self.hparams.readout_units ])


        self.built = True
    
    def call(self, inputs, training=False):
        f_ = inputs

        shape = tf.stack([f_['n_links'],self.hparams.link_state_dim-1], axis=0)

        link_state = tf.concat([
            tf.expand_dims(f_['capacities'],axis=1),
            tf.zeros(shape)
        ], axis=1)
        
        shape = tf.stack([f_['n_paths'],self.hparams.path_state_dim-1], axis=0)
        
        path_state = tf.concat([
            tf.expand_dims(f_['traffic'][0:f_["n_paths"]],axis=1),
            tf.zeros(shape)
        ], axis=1)

        # attention_state = tf.zeros([1, self.hparams.path_state_dim])
        
        # attention_state1 = tf.zeros([1, self.hparams.path_state_dim])

        attention_state2 = tf.zeros([1, self.hparams.path_state_dim])
        
        links = f_['links']
        paths = f_['paths']
        seqs=  f_['sequences']
        
                #relu1 = tf.keras.layers.ReLU(links, name='attention')
        W_link_r = self.hparams.link_c
        #self.hparams.link_state_dim
        W_link_c = self.hparams.link_state_dim
        W_path_r = self.hparams.path_c
        W_path_c = self.hparams.path_state_dim
        a = W_link_c+W_link_r
        W_link = tf.random.normal([W_link_r ,W_link_c], 0, 1, tf.float32, seed=None)
        print(W_link)
        W_path = tf.random.normal([W_path_r,W_path_c], 0, 1, tf.float32, seed=None)
        print(W_path)
        a_vector = tf.random.normal([a,1], 0, 1, tf.float32, seed=None)
        print(a_vector)

        
        for _ in range(self.hparams.T):
        
            h_ = tf.gather(link_state,links)
            
            #TODO move this to feature calculation
            ids=tf.stack([paths, seqs], axis=1)      
            #to find the max number of paths the link belongs to    
            max_len = tf.add(tf.reduce_max(seqs),1)

            shape = tf.stack([f_['n_paths'], max_len, self.hparams.link_state_dim])
            lens = tf.segment_sum(data=tf.ones_like(paths),
                                    segment_ids=paths)

            link_inputs = tf.scatter_nd(ids, h_, shape)
            #TODO move to tf.keras.RNN
            # relu = self.relu(attention_state)
            # relu1 = self.relu(attention_state1)
            relu2 = self.relu2(attention_state2)
            # e = tf.keras.layers.Multiply()([
            #     relu,
            #     path_state
            # ])
            # e1 = tf.keras.layers.Multiply()([
            #     relu1,
            #     link_state
            # ])
            outputs, path_state = tf.nn.dynamic_rnn(self.path_update,
                                                    link_inputs,
                                                    sequence_length=lens,
                                                    initial_state = path_state,
                                                    dtype=tf.float32)
            # rnn_layer = tf.keras.layers.RNN(self.path_update, return_sequences=True, return_state=True)
            # outputs, output_sts = rnn_layer(inputs = link_inputs, states=path_state)
            # e = tf.keras.layers.Multiply()(
            #     relu,
            #     path_state
            # ])

            ReLU_activ = tf.keras.layers.ReLU()
            m = tf.gather_nd(outputs,ids)
            # em = e*m
            # attention_state = em
            
            hp_ = tf.gather(path_state, paths)
            #hp_ = h_
            # print(tf.transpose(h_))
            # print('kkkkk')
            z_l = tf.linalg.matmul(W_link, tf.transpose(h_))
            # print(z_l)
            #z_p = tf.keras.layers.Multiply()([W_path, tf.transpose(hp_)])
            z_p = tf.linalg.matmul(W_path, tf.transpose(hp_))
            # print(z_p)
            z_l_p = tf.concat([z_l, z_p], axis=0)
            # print(z_l_p)
            # print('nnnnn')
            #a_z = tf.keras.layers.Multiply()([a_vector, z_l_p])
            a_z = z_l_p*a_vector
            a_z = tf.reduce_sum(a_z, 0)
            # print('a_z is down')
            # print(a_z)
            e_unnorm = ReLU_activ(a_z)
            # print('after relu')
            # print(e_unnorm)
            #layer1 = tf.keras.layers.LayerNormalization()
            #e_unnorm = tf.keras.utils.normalize(e_unnorm, axis=0, order=2)
            # print('normalized')
            # print(e_unnorm)
            # print(e_unnorm)
            e_unnorm = tf.math.l2_normalize(e_unnorm)
            em2 = e_unnorm*m
            
            # print('AFTERR')
            # print(m)
            # em = e*m
            # attention_state = em
            # em1 = e1*m
            # attention_state1 = em1 
            attention_state2 = em2
            #m = tf.keras.layers.Multiply()([m, e_unnorm])
            m = tf.unsorted_segment_sum(em2, links ,f_['n_links'])
            #Keras cell expects a list
            link_state,_ = self.edge_update(m, [link_state])

        if self.hparams.learn_embedding:
            r = self.readout(path_state,training=training)
            o = self.final(tf.concat([r,path_state], axis=1))
            
        else:
            r = self.readout(tf.stop_gradient(path_state),training=training)
            o = self.final(tf.concat([r, tf.stop_gradient(path_state)], axis=1) )
            
        return o  

def delay_model_fn(
   features, # This is batch_features from input_fn
   labels,   # This is batch_labrange
   mode,     # An instance of tf.estimator.ModeKeys
   params):  # Additional configuration

   
    model = RouteNet(params, output_units=2)
    model.build()

    predictions = model(features, training=mode==tf.estimator.ModeKeys.TRAIN)

    loc  = predictions[...,0] 
    c = np.log(np.expm1( np.float32(0.098) ))
    scale = tf.math.softplus(c + predictions[...,1]) + np.float32(1e-9)

    delay_prediction = loc
    jitter_prediction = scale**2


    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode, 
            predictions={'delay':delay_prediction, 'jitter':jitter_prediction}
            )

    with tf.name_scope('heteroscedastic_loss'):
        x=features
        y=labels

        n=x['packets']-y['drops']
        _2sigma = np.float32(2.0)*scale**2
        nll = n*y['jitter']/_2sigma + n*tf.math.squared_difference(y['delay'], loc)/_2sigma + n*tf.math.log(scale)
        loss = tf.reduce_sum(nll)/np.float32(1e6)

    regularization_loss = sum(model.losses)
    total_loss = loss + regularization_loss

    if mode == tf.estimator.ModeKeys.EVAL:
        return tf.estimator.EstimatorSpec(
            mode,loss=loss,
            eval_metric_ops={
                'label/mean/delay':tf.metrics.mean(labels['delay']),
                'label/mean/jitter':tf.metrics.mean(labels['jitter']),
                'prediction/mean/delay': tf.metrics.mean(delay_prediction),
                'prediction/mean/jitter': tf.metrics.mean(jitter_prediction),
                'mae/delay':tf.metrics.mean_absolute_error(labels['delay'], delay_prediction),
                'mae/jitter':tf.metrics.mean_absolute_error(labels['jitter'], jitter_prediction),
                'rho/delay':tf.contrib.metrics.streaming_pearson_correlation(labels=labels['delay'],predictions=delay_prediction),
                'rho/jitter':tf.contrib.metrics.streaming_pearson_correlation(labels=labels['jitter'],predictions=jitter_prediction)
            }
        )
    
    assert mode == tf.estimator.ModeKeys.TRAIN


    trainables = model.variables
    grads = tf.gradients(total_loss, trainables)
    grad_var_pairs = zip(grads, trainables)

    summaries = [tf.summary.histogram(var.op.name, var) for var in trainables]
    summaries += [tf.summary.histogram(g.op.name, g) for g in grads if g is not None]

    decayed_lr = tf.train.exponential_decay(params.learning_rate,
                                            tf.train.get_global_step(), 50000,
                                            0.9, staircase=True)

    optimizer=tf.train.AdamOptimizer(decayed_lr)
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        train_op = optimizer.apply_gradients(grad_var_pairs,
            global_step=tf.train.get_global_step())

    return tf.estimator.EstimatorSpec(mode, 
            loss=total_loss, 
            train_op=train_op,
        )


def drop_model_fn(
   features, # This is batch_features from input_fn
   labels,   # This is batch_labrange
   mode,     # An instance of tf.estimator.ModeKeys
   params):  # Additional configuration

   
    model = RouteNet(params, output_units=1, final_activation=None)
    model.build()

    logits = model(features, training=mode==tf.estimator.ModeKeys.TRAIN)
    logits = tf.squeeze(logits)
    predictions = tf.math.sigmoid(logits)
    
    ###new addition
    loc = predictions[...,0]

    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode, 
            predictions={'drops':predictions, 'logits':logits}
            )

    with tf.name_scope('binomial_loss'):
        x=features
        y=labels

        loss_ratio = y['drops']/x['packets']
        # Binomial negative Log-likelihood
        loss = tf.reduce_sum(x['packets']*tf.nn.sigmoid_cross_entropy_with_logits(
            labels = loss_ratio,
            logits = logits
        ))/np.float32(1e5)

    regularization_loss = sum(model.losses)
    total_loss = loss + regularization_loss
    tf.summary.scalar('regularization_loss', regularization_loss)


    if mode == tf.estimator.ModeKeys.EVAL:
        return tf.estimator.EstimatorSpec(
            mode,loss=loss,
            eval_metric_ops={
                'label/mean/drops':tf.metrics.mean(loss_ratio),
                'prediction/mean/drops': tf.metrics.mean(predictions),
                'mae/drops':tf.metrics.mean_absolute_error(loss_ratio, predictions),
                'rho/drops':tf.contrib.metrics.streaming_pearson_correlation(labels=loss_ratio,predictions=predictions),
                'mre':tf.compat.v1.metrics.mean_relative_error(loss_ratio, predictions, normalizer=loss_ratio)
            }
        )
    
    assert mode == tf.estimator.ModeKeys.TRAIN


    trainables = model.trainable_variables
    grads = tf.gradients(total_loss, trainables)
    grad_var_pairs = zip(grads, trainables)

    summaries = [tf.summary.histogram(var.op.name, var) for var in trainables]
    summaries += [tf.summary.histogram(g.op.name, g) for g in grads if g is not None]

    decayed_lr = tf.train.exponential_decay(params.learning_rate,
                                            tf.train.get_global_step(), 50000,
                                            0.9, staircase=True)
    # TODO use decay !
    optimizer=tf.train.AdamOptimizer(decayed_lr)
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        train_op = optimizer.apply_gradients(grad_var_pairs,
            global_step=tf.train.get_global_step())
    #could track total loss here

    return tf.estimator.EstimatorSpec(mode, 
            loss=total_loss, 
            train_op=train_op,
        )

def scale_fn(k, val):
    '''Scales given feature
    Args:
        k: key
        val: tensor value
    '''

    if k == 'traffic':
        return (val-0.18)/.15
    if k == 'capacities':
        return val/10.0

    return val

def parse(serialized, target=None, normalize=True):
    '''
    Target is the name of predicted variable-deprecated
    '''
    with tf.device("/cpu:0"):
        with tf.name_scope('parse'):    
            #TODO add feature spec class
            features = tf.io.parse_single_example(
                serialized,
                features={
                    'traffic':tf.VarLenFeature(tf.float32),
                    'delay':tf.VarLenFeature(tf.float32),
                    'logdelay':tf.VarLenFeature(tf.float32),
                    'jitter':tf.VarLenFeature(tf.float32),
                    'drops':tf.VarLenFeature(tf.float32),
                    'packets':tf.VarLenFeature(tf.float32),
                    'capacities':tf.VarLenFeature(tf.float32),
                    'links':tf.VarLenFeature(tf.int64),
                    'paths':tf.VarLenFeature(tf.int64),
                    'sequences':tf.VarLenFeature(tf.int64),
                    'n_links':tf.FixedLenFeature([],tf.int64), 
                    'n_paths':tf.FixedLenFeature([],tf.int64),
                    'n_total':tf.FixedLenFeature([],tf.int64)
                })
            for k in ['traffic','delay','logdelay','jitter','drops','packets','capacities','links','paths','sequences']:
                features[k] = tf.sparse.to_dense( features[k] )
                if normalize:
                    features[k] = scale_fn(k, features[k])


    #return {k:v for k,v in features.items() if k is not target },features[target]
    return features

def cummax(alist, extractor):
    with tf.name_scope('cummax'):
        maxes = [tf.reduce_max( extractor(v) ) + 1 for v in alist ]
        cummaxes = [tf.zeros_like(maxes[0])]
        for i in range(len(maxes)-1):
            cummaxes.append( tf.math.add_n(maxes[0:i+1]))
        
    return cummaxes

def transformation_func(it, batch_size=32):
    with tf.name_scope("transformation_func"):
        vs = [it.get_next() for _ in range(batch_size)]
        
        links_cummax = cummax(vs,lambda v:v['links'] )
        paths_cummax = cummax(vs,lambda v:v['paths'] )
        
        tensors = ({
                'traffic':tf.concat([v['traffic'] for v in vs], axis=0),
                'capacities': tf.concat([v['capacities'] for v in vs], axis=0),
                'sequences':tf.concat([v['sequences'] for v in vs], axis=0),
                'packets':tf.concat([v['packets'] for v in vs], axis=0),
                'links':tf.concat([v['links'] + m for v,m in zip(vs, links_cummax) ], axis=0),
                'paths':tf.concat([v['paths'] + m for v,m in zip(vs, paths_cummax) ], axis=0),
                'n_links':tf.math.add_n([v['n_links'] for v in vs]),
                'n_paths':tf.math.add_n([v['n_paths'] for v in vs]),
                'n_total':tf.math.add_n([v['n_total'] for v in vs])
            },   {
                'delay' : tf.concat([v['delay'] for v in vs], axis=0),
                'logdelay' : tf.concat([v['logdelay'] for v in vs], axis=0),
                'drops' : tf.concat([v['drops'] for v in vs], axis=0),
                'jitter' : tf.concat([v['jitter'] for v in vs], axis=0),
                }
            )
    
    return tensors

def tfrecord_input_fn(filenames,hparams,shuffle_buf=1000, target='delay'):
    
    files = tf.data.Dataset.from_tensor_slices(filenames)
    files = files.shuffle(len(filenames))

    ds = files.apply(tf.data.experimental.parallel_interleave(
        tf.data.TFRecordDataset, cycle_length=4))

    if shuffle_buf:
        ds = ds.apply(tf.data.experimental.shuffle_and_repeat(shuffle_buf))
    else :
        # sample 10 % for evaluation because it is time consuming
        ds = ds.filter(lambda x: tf.random_uniform(shape=())< 0.1)

    ds = ds.map(lambda buf:parse(buf,target), 
        num_parallel_calls=2)
    ds=ds.prefetch(10)

    it =ds.make_one_shot_iterator()
    sample = transformation_func(it,hparams.batch_size)
    

    return sample

def serving_input_receiver_fn():
    """
    This is used to define inputs to serve the model.
    returns: ServingInputReceiver
    """
    receiver_tensors = {
        'capacities': tf.placeholder(tf.float32, [None]),
        'traffic': tf.placeholder(tf.float32, [None]),
        'links': tf.placeholder(tf.int32, [None]),
        'paths': tf.placeholder(tf.int32, [None]),
        'sequences': tf.placeholder(tf.int32, [None]),
        'n_links': tf.placeholder(tf.int32, []),
        'n_paths':tf.placeholder(tf.int32, []),
    }

    # Convert give inputs to adjust to the model.
    features = {k: scale_fn(k,v) for k,v in receiver_tensors.items() }
    return tf.estimator.export.ServingInputReceiver(receiver_tensors=receiver_tensors,
                                                    features=features)




os.environ["CUDA_VISIBLE_DEVICES"]="0"

from tensorflow.core.framework.summary_pb2 import Summary
from tensorflow.python.training import training_util
from tensorflow.python.training.session_run_hook import SessionRunArgs
import pandas as pd
import csv
import time

class evaluate_and_save_to_csv_hook(tf.train.SessionRunHook):
    #evaluate an estimator every n steps and save to a csv
    def __init__(self, estimator, steps_per_save, csv_directory, input_fn, eval_steps):
        self._estimator = estimator
        self.stps = steps_per_save
        self.csv_dir = csv_directory
        self.input_fn = input_fn
        self.evl_stps = eval_steps
        self.start_time = time.time()

    def after_create_session(self, session, coord):
        self._global_step_tensor = training_util._get_or_create_global_step_read()

    def begin(self):
        pass

    def before_run(self, run_context):  # pylint: disable=unused-argument
        return SessionRunArgs(self._global_step_tensor)

    def after_run(self, run_context, run_values):
        if run_values.results == 0:
            _time_check_start = time.time()
            _dict = self._estimator.evaluate(input_fn=self.input_fn, steps=2)
            _time_check_end = time.time()
            _dict['step'] = run_values.results
            _dict['evaltime'] = _time_check_end-_time_check_start
            _dict['time'] = _time_check_end-self.start_time
            with open(self.csv_dir, "w") as csv_file:
                writer = csv.writer(csv_file, delimiter=',')
                _ = [i for i in _dict.keys()]
                writer.writerow(_)
                
        if run_values.results%self.stps == 0:
            _time_check_start = time.time()
            _dict = self._estimator.evaluate(input_fn=self.input_fn, steps=self.evl_stps)
            _time_check_end = time.time()
            _dict['step'] = run_values.results
            _dict['evaltime'] = _time_check_end-_time_check_start
            _dict['time'] = _time_check_end-self.start_time
            with open(self.csv_dir, "a") as csv_file:
                writer = csv.writer(csv_file, delimiter=',')
                _ = [_dict[i] for i in _dict.keys()]
                writer.writerow(_)
                #can trigger something here

class loss_logger(tf.train.SessionRunHook):
    def __init__(self, steps_per_save, loss_directory):
        self.csv_dir = loss_directory
        self.stps = steps_per_save
        self.start_time = time.time()

    def after_create_session(self, session, coord):
        self._global_step_tensor = training_util._get_or_create_global_step_read()
        

    def begin(self):
        pass

    def before_run(self, run_context):  # pylint: disable=unused-argument
        return SessionRunArgs(self._global_step_tensor)

    def after_run(self, run_context, run_values):
        step = run_values.results
        if step == 0:
            with open(self.csv_dir, "w") as csv_file:
                writer = csv.writer(csv_file, delimiter=',')
                _ = ['step', 'loss', 'time']
                writer.writerow(_)
                
        if step%self.stps == 0:
            losses = run_context.session.graph.get_collection("losses")
            loss = run_context.session.run(losses)[0]
            with open(self.csv_dir, "a") as csv_file:
                writer = csv.writer(csv_file, delimiter=',')
                _time = time.time()-self.start_time
                _ = [step, loss, _time]
                writer.writerow(_)
                #can trigger something here

hparams = tf.contrib.training.HParams(
    node_count=14,
    link_state_dim=32, 
    #[4, 8, 16, 32, 64]
    path_state_dim=32,
    #[2, 4, 8, 16, 32, 64]
    T=8,
    readout_units=8,
    learning_rate=0.001,
    #[.001, .01, .05]
    batch_size=32,
    #[8, 16, 32, 64]
    dropout_rate=0.5,
    #[.5] leave
    l2=0.1,
    #regulirization constants
    #[.05, .1, .2]
    l2_2=0.01,
    #[.005, .01, .02]
    learn_embedding=True, # If false, only the readout is trained
    readout_layers=2, # number of hidden layers in readout model
    #[2, 3, 4]
    link_c = 32,
    path_c = 32
)

##Save Folders
#saved stuff
SaveMain = '/root/to_save_reduced_lite'
Logs = SaveMain+'/Logs'
Models = SaveMain+'/Models/'
Train_Logs = Logs+'/Train_Logs/'



estimator = tf.estimator.Estimator( 
        model_fn = drop_model_fn, 
        params=hparams,
        model_dir=Models+'model_gat1_simplegat_lastnight_em2onlyNormLLL'
        )

tf.logging.set_verbosity('INFO') 
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

estimator.train(input_fn=lambda:tfrecord_input_fn(full_train, hparams, shuffle_buf=10000, target='delay'), 
                max_steps=5000,
                hooks = [
                    evaluate_and_save_to_csv_hook(
                        estimator, 
                        steps_per_save=100, 
                        csv_directory=Train_Logs+'val.csv', 
                        input_fn = lambda:tfrecord_input_fn(full_valid, hparams, shuffle_buf=10000, target='delay'), 
                        eval_steps=len(full_valid)*125//32
                        ), 
                    loss_logger(steps_per_save=100, loss_directory=Train_Logs+'loss.csv')
                    ]
                 )
estimator.evaluate(input_fn=lambda:tfrecord_input_fn(full_test2, hparams, shuffle_buf=10000, target='delay'), 
                 steps=3398)