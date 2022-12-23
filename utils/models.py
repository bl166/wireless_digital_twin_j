import numpy as np
import pandas as pd
import spektral as spk
from copy import deepcopy
import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import tensorflow as tf
#print(tf.test.gpu_device_name())


class PlanNet(tf.keras.Model):
    def __init__(self, hparams, output_units=1, final_activation=None, train_on='delay'):
        super().__init__()
        self.hparams = deepcopy(hparams)
        self.output_units = output_units
        self.final_activation = final_activation
        self.train_on = train_on

    def _build_dims_helper(self):
        path_update_in = [None, self.hparams['link_state_dim']+self.hparams['node_state_dim']]
        edge_update_in = [None, self.hparams['link_state_dim']+self.hparams['node_state_dim']+self.hparams['path_state_dim']]
        node_update_in = self.hparams['node_state_dim']
        return path_update_in, edge_update_in, node_update_in
        
    def build(self, input_shape=None):
        del input_shape
        path_update_in, edge_update_in, node_update_in = self._build_dims_helper()
        self._build_dims_helper()

        #state updaters 
        # path
        self.path_update = tf.keras.layers.GRUCell(self.hparams['path_state_dim'], name="path_update")
        self.path_update.build(input_shape = path_update_in)
        
        # edge
        self.edge_update = tf.keras.models.Sequential(name="edge_update")
        if 'edgeMLPLayerSizes' not in self.hparams:
            self.hparams['edgeMLPLayerSizes'] = [self.hparams['link_state_dim']] * 5
        for i, edgemlp_units in enumerate(self.hparams['readoutLayerSizes']):
            self.edge_update.add(tf.keras.layers.Dense(edgemlp_units))
        self.edge_update.build(input_shape = edge_update_in)
        
        # node
        if node_update_in:
            self.node_update = spk.layers.ECCConv(node_update_in)

        #readout-final
        self.readout = tf.keras.models.Sequential(name='readout')
        if 'readoutLayerSizes' not in self.hparams:
            self.hparams['readoutLayerSizes'] = [self.hparams['readout_units']] * self.hparams['readout_layers']
        for i, readout_units in enumerate(self.hparams['readoutLayerSizes']):
            self.readout.add(tf.keras.layers.Dense(
                readout_units, 
                activation = tf.nn.selu,
                kernel_regularizer = tf.keras.regularizers.L2(self.hparams['l2'])
            ))
            self.readout.add(tf.keras.layers.Dropout(rate = self.hparams['dropout_rate']))
        self.readout.build(input_shape = [None, self.hparams['path_state_dim']])

        self.final = tf.keras.layers.Dense(self.output_units, 
                kernel_regularizer=tf.keras.regularizers.L2(self.hparams['l2_2']),
                activation = self.final_activation )
        self.final.build(input_shape = [None, self.hparams['path_state_dim'] + self.hparams['readoutLayerSizes'][0] ])
        self.built = True

    
    def call(self, inputs, training=False):
        #call == v ==
        f_ = inputs

        #state init
        shape = tf.stack([f_['n_links'],self.hparams['link_state_dim']-1], axis=0)
        link_state = tf.concat([
            tf.expand_dims(f_['link_init'],axis=1),
            tf.zeros(shape)
        ], axis=1)

        shape = tf.stack([f_['n_nodes'],self.hparams['node_state_dim']-1], axis=0)
        node_state = tf.concat([
            tf.expand_dims(f_['node_init'],axis=1),
            tf.zeros(shape)
        ], axis=1)

        shape = tf.stack([f_['n_paths'],self.hparams['path_state_dim']-2], axis=0)
        path_state = tf.concat([
            tf.expand_dims(f_['path_init'][0],axis=1),
            tf.expand_dims(f_['path_init'][1],axis=1),
            tf.zeros(shape)
        ], axis=1)

        #pull for both
        paths   = f_['paths_to_links']
        seqs    = f_['sequences_paths_links']
        n_paths = f_['n_paths']
        
        for _ in range(self.hparams['T']):
        #stuff for both
            ids=tf.stack([paths, seqs], axis=1)
            max_len = tf.reduce_max(seqs)+1
            lens = tf.math.segment_sum(data=tf.ones_like(paths), segment_ids=paths)
            
            #link stuff
            h_ = tf.gather(link_state,f_['links_to_paths'])

            shape = tf.stack([n_paths, max_len, self.hparams['link_state_dim']])
            link_inputs = tf.scatter_nd(ids, h_, shape)
            
            #node stuff
            h_ = tf.gather(link_state,f_['nodes_to_paths'])

            shape = tf.stack([n_paths, max_len, self.hparams['node_state_dim']])
            node_inputs = tf.scatter_nd(ids, h_, shape)
            
            x_inputs = tf.concat([link_inputs, node_inputs], axis=2)
            
            #updating path_state
            outputs, path_state = tf.compat.v1.nn.dynamic_rnn(
                cell            = self.path_update,
                inputs          = x_inputs,
                sequence_length = lens,
                initial_state   = path_state,
                dtype           = tf.float32
            )
            
            m = tf.gather_nd(outputs,ids)
            m = tf.math.unsorted_segment_sum(m, f_['links_to_paths'] ,f_['n_links'])

            #fitting nodes to links
            h_ = tf.gather(node_state,f_['links_to_nodes'])

            _con = tf.concat([h_, link_state, m], axis=1)
            link_state = self.edge_update(_con)
            
            #node update
            node_state = self.node_update((node_state, f_['adjacency_matrix'], link_state))

        #readout
        if self.hparams['learn_embedding']:
            r = self.readout(path_state,training=training)
            o = self.final(tf.concat([r,path_state], axis=1))
        else:
            r = self.readout(tf.stop_gradient(path_state),training=training)
            o = self.final(tf.concat([r, tf.stop_gradient(path_state)], axis=1) )

        return o
    

    def train_step(self, data):
        features, labels = data
        print(type(features), type(labels))
        
        with tf.GradientTape() as tape:
            predictions = self(features, training=True)
            print('train_step | pred:', tf.math.reduce_any(tf.math.is_nan(predictions)), predictions.shape)
            loc  = predictions[...,0]
            kpi_prediction = loc
            loss = tf.keras.metrics.mean_squared_error(labels[self.train_on], loc)
            print('train_step | loss:', tf.math.reduce_any(tf.math.is_nan(loss)), tf.math.reduce_mean(loss))

            regularization_loss = sum(self.losses)
            total_loss = loss + regularization_loss
            
        gradients = tape.gradient(total_loss, self.trainable_variables)
        
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
        ret = {
            'loss':loss,
            f'label/mean/{self.train_on}':tf.math.reduce_mean(labels[self.train_on]),
            f'prediction/mean/{self.train_on}': tf.math.reduce_mean(kpi_prediction)
            }
        return ret
    

    def test_step(self, data):
        features, labels = data
        print(features)
        
        with tf.GradientTape() as tape:
            predictions = self(features, training=False)
            loc  = predictions[...,0]
            kpi_prediction = loc
            loss = tf.keras.metrics.mean_squared_error(labels[self.train_on], loc)

            regularization_loss = sum(self.losses)
            total_loss = loss + regularization_loss
            
        ret = {
            'loss':loss,
            f'label/mean/{self.train_on}':tf.math.reduce_mean(labels[self.train_on]),
            f'prediction/mean/{self.train_on}': tf.math.reduce_mean(kpi_prediction)
            }
        return ret
    
    
    
class RouteNet(PlanNet):   
    def __init__(self, hparams, output_units=1, final_activation=None, train_on="delay"):
        super().__init__(hparams, output_units, final_activation, train_on)
        self.hparams['node_state_dim'] = 0
    
    def call(self, inputs, training=False):
        #call == v ==
        f_ = inputs

        #state init
        shape = tf.stack([f_['n_links'],self.hparams['link_state_dim']-1], axis=0)
        link_state = tf.concat([
            tf.expand_dims(f_['link_init'],axis=1),
            tf.zeros(shape)
        ], axis=1)

#         shape = tf.stack([f_['n_nodes'],self.hparams['node_state_dim']-1], axis=0)
#         node_state = tf.concat([
#             tf.expand_dims(f_['node_init'],axis=1),
#             tf.zeros(shape)
#         ], axis=1)

        shape = tf.stack([f_['n_paths'],self.hparams['path_state_dim']-2], axis=0)
        path_state = tf.concat([
            tf.expand_dims(f_['path_init'][0],axis=1),
            tf.expand_dims(f_['path_init'][1],axis=1),
            tf.zeros(shape)
        ], axis=1)

        #pull for both
        paths   = f_['paths_to_links']
        seqs    = f_['sequences_paths_links']
        n_paths = f_['n_paths']
        
        for _ in range(self.hparams['T']):
        #stuff for both
            ids=tf.stack([paths, seqs], axis=1)
            max_len = tf.reduce_max(seqs)+1
            lens = tf.math.segment_sum(data=tf.ones_like(paths), segment_ids=paths)
            
            #link stuff
            h_ = tf.gather(link_state,f_['links_to_paths'])

            shape = tf.stack([n_paths, max_len, self.hparams['link_state_dim']])
            link_inputs = tf.scatter_nd(ids, h_, shape)
             
            #updating path_state
            outputs, path_state = tf.compat.v1.nn.dynamic_rnn(
                cell            = self.path_update,
                inputs          = link_inputs,
                sequence_length = lens,
                initial_state   = path_state,
                dtype           = tf.float32
            )
            
            m = tf.gather_nd(outputs,ids)
            m = tf.math.unsorted_segment_sum(m, f_['links_to_paths'] ,f_['n_links'])

#             #fitting nodes to links
#             h_ = tf.gather(node_state, f_['links_to_nodes'])
#             _con = tf.concat([h_, link_state, m], axis=1)
            _con = tf.concat([link_state, m], axis=1)
            link_state = self.edge_update(_con)
            
#             #node update
#             node_state = self.node_update((node_state, f_['adjacency_matrix'], link_state))

        #readout
        if self.hparams['learn_embedding']:
            r = self.readout(path_state,training=training)
            o = self.final(tf.concat([r,path_state], axis=1))
        else:
            r = self.readout(tf.stop_gradient(path_state),training=training)
            o = self.final(tf.concat([r, tf.stop_gradient(path_state)], axis=1) )

        return o