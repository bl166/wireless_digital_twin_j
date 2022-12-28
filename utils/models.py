import numpy as np
import pandas as pd
import spektral as spk
from copy import deepcopy
import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

import tensorflow as tf
#print(tf.test.gpu_device_name())


class PlanNet(tf.keras.Model):
    def __init__(self, hparams, output_units=1, final_activation=None, train_on=['delay']):
        super().__init__()
        self.hparams          = deepcopy(hparams)
        self.output_units     = output_units
        self.final_activation = final_activation
        self.train_on         = train_on[0]
        assert output_units == len(train_on)

    def _build_dims_helper(self):
        path_update_in = [None, self.hparams['link_state_dim']+self.hparams['node_state_dim']]
        edge_update_in = [None, self.hparams['link_state_dim']+self.hparams['node_state_dim']+self.hparams['path_state_dim']]
        node_update_in = self.hparams['node_state_dim']
        return path_update_in, edge_update_in, node_update_in
        
    def build(self):
        path_update_in, edge_update_in, node_update_in = self._build_dims_helper()

        #state updaters - path
        self.path_update = tf.keras.layers.RNN(
            tf.keras.layers.GRUCell(int(self.hparams['path_state_dim'])),
            return_sequences = True,
            return_state     = True,
            dtype            = tf.float32, 
            name             = 'path_update'
        )
        
        # edge
        self.edge_update = tf.keras.models.Sequential(name='edge_update')
        if 'edgeMLPLayerSizes' not in self.hparams:
            self.hparams['edgeMLPLayerSizes'] = [self.hparams['link_state_dim']] * 4
        for edgemlp_units in self.hparams['edgeMLPLayerSizes']:
            self.edge_update.add(tf.keras.layers.Dense(
                edgemlp_units,
                activation         = tf.nn.relu,
                kernel_regularizer = tf.keras.regularizers.L2(self.hparams['l2'])
            ))
        self.edge_update.add(tf.keras.layers.Dense(self.hparams['link_state_dim']))
 
        # node
        if node_update_in:
            self.node_update = spk.layers.GCNConv(node_update_in)

        #readout-final
        self.readout = tf.keras.models.Sequential(name='readout')
        if 'readoutLayerSizes' not in self.hparams:
            self.hparams['readoutLayerSizes'] = [self.hparams['readout_units']] * self.hparams['readout_layers']
        for readout_units in self.hparams['readoutLayerSizes']:
            self.readout.add(tf.keras.layers.Dense(
                readout_units, 
                activation         = tf.nn.relu,
                kernel_regularizer = tf.keras.regularizers.L2(self.hparams['l2'])
            ))
            self.readout.add(tf.keras.layers.Dropout(rate = self.hparams['dropout_rate']))

        self.final = tf.keras.layers.Dense(
            self.output_units, 
            kernel_regularizer = tf.keras.regularizers.L2(self.hparams['l2_2']),
            activation         = self.final_activation 
        )
        
        self.edge_update.build(input_shape = edge_update_in)
        self.readout.build(input_shape = [None, self.hparams['path_state_dim']])
        self.final.build(input_shape = [None, self.hparams['path_state_dim']+readout_units])
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

            ###################### PATH STATE #################################
            
            ids=tf.stack([paths, seqs], axis=1)
            max_len = tf.reduce_max(seqs)+1
            lens = tf.math.segment_sum(data=tf.ones_like(paths), segment_ids=paths)
            
            # Collect link states of all the links included in all the paths 
            h_ = tf.gather(link_state,f_['links_to_paths'])
            shape = tf.stack([n_paths, max_len, self.hparams['link_state_dim']])
            link_inputs = tf.scatter_nd(ids, h_, shape)
            
            # Collect node states of all the nodes included in all the paths 
            h1_ = tf.gather(node_state,f_['nodes_to_paths'])
            shape = tf.stack([n_paths, max_len, self.hparams['node_state_dim']])
            node_inputs = tf.scatter_nd(ids, h1_, shape)
            
            # Concatenate link state with corresponding source node's state
            x_inputs = tf.concat([link_inputs, node_inputs], axis=2)
            
            # Update path state
            outputs, path_state = self.path_update(
                inputs        = x_inputs, 
                initial_state = path_state
            )
    
            ###################### LINK STATE #################################       
            m = tf.gather_nd(outputs,ids)
            m = tf.math.unsorted_segment_sum(
                data         = m, 
                segment_ids  = f_['links_to_paths'],
                num_segments = f_['n_links']
            )
            #fitting nodes to links
            h2_ = tf.gather(node_state,f_['links_to_nodes'])
            _con = tf.concat([h2_, link_state, m], axis=1)
            link_state = self.edge_update(_con)
            
            ###################### NODE STATE ################################# 
            h3_ = tf.gather(link_state, f_['nodes_to_links'])
            agg = tf.math.unsorted_segment_sum(
                data         = h3_, 
                segment_ids  = f_['links_to_nodes'],
                num_segments = f_['n_nodes']
            )
            _con2 = tf.concat([node_state, agg], axis=1)
            node_state = self.node_update((_con2,f_['laplacian_matrix']))
            
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

            regularization_loss = tf.math.reduce_sum(self.losses)
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

            regularization_loss = tf.math.reduce_sum(self.losses)
            total_loss = loss + regularization_loss
            
        ret = {
            'loss':loss,
            f'label/mean/{self.train_on}':tf.math.reduce_mean(labels[self.train_on]),
            f'prediction/mean/{self.train_on}': tf.math.reduce_mean(kpi_prediction)
            }
        return ret
    
    
    
class RouteNet(PlanNet):   
    def __init__(self, hparams, output_units=1, final_activation=None, train_on='delay'):
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
            ids=tf.stack([paths, seqs], axis=1)
            max_len = tf.reduce_max(seqs)+1
            lens = tf.math.segment_sum(data=tf.ones_like(paths), segment_ids=paths)
            
            #link stuff
            h_ = tf.gather(link_state,f_['links_to_paths'])

            shape = tf.stack([n_paths, max_len, self.hparams['link_state_dim']])
            link_inputs = tf.scatter_nd(ids, h_, shape)

            #updating path_state
            outputs, path_state = self.path_update(
                inputs        = link_inputs, 
                initial_state = path_state
            )
            
            m = tf.gather_nd(outputs,ids)
            m = tf.math.unsorted_segment_sum(m, f_['links_to_paths'] ,f_['n_links'])

            #fitting nodes to links
            _con = tf.concat([link_state, m], axis=1)
            link_state = self.edge_update(_con)
            
        #readout
        if self.hparams['learn_embedding']:
            r = self.readout(path_state,training=training)
            o = self.final(tf.concat([r, path_state], axis=1))
        else:
            r = self.readout(tf.stop_gradient(path_state),training=training)
            o = self.final(tf.concat([r, tf.stop_gradient(path_state)], axis=1) )

        return o