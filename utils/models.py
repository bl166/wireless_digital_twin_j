import numpy as np
import pandas as pd
from copy import deepcopy
import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

import tensorflow as tf
#print(tf.test.gpu_device_name())
from tensorflow.keras import backend as K
from tensorflow.keras import layers,initializers
from tensorflow.keras.models import Model
import spektral as spk

from .graph_convs import GCNConvBN, ECCConvBN
    

class PlanNet(tf.keras.Model):
    def __init__(self, hparams, output_units=1, batch_norm = False, batching=False, sharing=True,
                 final_activation=None, train_on=['delay'], loss='mse', **argw):
        super().__init__()
        self.hparams          = deepcopy(hparams)
        self.final_activation = final_activation
        self.train_on         = train_on#[0]
        self.output_units     = len(self.train_on) #output_units
        #assert output_units == self.output_units
        
        # whether enable param share
        self.share = sharing
        if self.share:
            self.build_path_layers = self.build_path_layers_share
            self.build_edge_layers = self.build_edge_layers_share
            self.build_node_layers = self.build_node_layers_share
        else:
            self.build_path_layers = self.build_path_layers_noshare
            self.build_edge_layers = self.build_edge_layers_noshare
            self.build_node_layers = self.build_node_layers_noshare        
        
        # set batch
        if batching or ('batching' in hparams and hparams['batching']):
            self.batch_size = self.hparams['batch_size']
            self.use_norm = batch_norm
        else:
            self.batch_size = 0
            self.use_norm = False
            
        # set loss
        self._get_loss_func(loss)
        
            
    def _get_loss_func(self, loss):
        if loss.lower()=='bce': #'-bi' in self.train_on:
            self.loss_func = lambda target, output: tf.reduce_mean(
                K.binary_crossentropy(target, output, from_logits=False), 
                axis=0
            )
        elif loss.lower()=='mse':
            self.loss_func = lambda target, output: tf.reduce_mean(
                tf.math.square(output - target), 
                axis=0
            )
        elif loss.lower()=='logcosh':
            self.loss_func = lambda target, output: tf.keras.losses.logcosh(
                target, output
            )
        elif loss.lower()=='mae':
            self.loss_func = lambda target, output: tf.reduce_mean(
                tf.abs(output - target), 
                axis=0
            )
        elif loss.lower()=='msle':
            self.loss_func = lambda target, output: tf.reduce_mean(
                tf.math.square(tf.math.log(target + 1.) - tf.math.log(output + 1.)), 
                axis=0
            )
        else:
            raise
            
    def _build_dims_helper(self):
        path_update_in = [self.hparams['path_state_dim'], self.hparams['link_state_dim']+self.hparams['node_state_dim']]
        edge_update_in = self.hparams['link_state_dim']+self.hparams['node_state_dim']+self.hparams['path_state_dim']
        node_update_in = self.hparams['node_state_dim']
        return path_update_in, edge_update_in, node_update_in
    
    def build(self):
        path_update_in, edge_update_in, node_update_in = self._build_dims_helper()
        self.build_path_layers(path_update_in)
        self.build_edge_layers(edge_update_in)
        if self.hparams['node_state_dim']:
            self.build_node_layers(node_update_in)      
        self.build_readout_layers() #readout-final
        self.built = True
        
        
    def _build_path_layers_helper(self, update_in, ti=None):
        path_update = layers.RNN(
            layers.GRUCell(int(self.hparams['path_state_dim'])),
            return_sequences = True,
            return_state     = True,
            dtype            = tf.float32, 
            name             = f'path_update' + (f'_t{ti}' if ti is not None else '')
        )
        path_update.build(tf.TensorShape([None,*update_in]))
        return path_update
    
    def _build_edge_layers_helper(self, update_in, ti=None):
        edge_update = tf.keras.models.Sequential(
            name='edge_update' + (f'_t{ti}' if ti is not None else '')
        )
        if 'edgeMLPLayerSizes' not in self.hparams:
            self.hparams['edgeMLPLayerSizes'] = [self.hparams['link_state_dim']] * 4
        self.hparams['edgeMLPLayerSizes'] += [self.hparams['link_state_dim']] # append last unit
        for edgemlp_units in self.hparams['edgeMLPLayerSizes']:
            edge_update.add(
                layers.Dense(edgemlp_units, kernel_regularizer = tf.keras.regularizers.L2(self.hparams['l2']) )
            )
            if self.use_norm:
                edge_update.add(layers.LayerNormalization())
            edge_update.add(layers.ReLU())
        edge_update.build([None, update_in])
        return edge_update
        
    def _build_node_layers_helper(self, update_in, ti=None):
        self.support = 'adjacency_matrix'
        node_update = ECCConvBN(
            self.hparams['node_state_dim'],
            batch_norm = self.use_norm, 
            dropout    = 0, #self.hparams['dropout_rate'],
            activation = tf.nn.relu,
            name=f'node_update' + (f'_t{ti}' if ti is not None else '')
        )   
        node_update.build(input_shape = [None, [update_in]])  
        return node_update
        
    # -----------------------------------------------
    # ---- not to share parameters across layers ----
    # -----------------------------------------------
    def build_path_layers_noshare(self, update_in):
        self.path_update = {}
        for ti in range(self.hparams['T']):
            self.path_update[str(ti)] = self._build_path_layers_helper(update_in, ti)
        
    def build_edge_layers_noshare(self, update_in):
        self.edge_update = {}
        for ti in range(self.hparams['T']-1):
            self.edge_update[str(ti)] = self._build_edge_layers_helper(update_in, ti)
        
    def build_node_layers_noshare(self, update_in):  
        self.node_update = {}
        for ti in range(self.hparams['T']):
            self.edge_update[str(ti)] = self._build_node_layers_helper(update_in, ti)
            
    # -----------------------------------------
    # ---- share parameters across layers -----
    # -----------------------------------------
    def build_path_layers_share(self, update_in):
        self.path_update = self._build_path_layers_helper(update_in)
                
    def build_edge_layers_share(self, update_in):
        self.edge_update = self._build_edge_layers_helper(update_in)
        
    def build_node_layers_share(self, update_in):  
        self.node_update = self._build_node_layers_helper(update_in)
        
    def build_readout_layers(self):
        self.readout = tf.keras.models.Sequential(name='readout')
        if 'readoutLayerSizes' not in self.hparams:
            self.hparams['readoutLayerSizes'] = [self.hparams['readout_units']] * self.hparams['readout_layers']
        for readout_units in self.hparams['readoutLayerSizes']:
            self.readout.add(
                layers.Dense(readout_units, kernel_regularizer = tf.keras.regularizers.L2(self.hparams['l2_2']))
            )
            if self.use_norm:
                self.readout.add(layers.LayerNormalization())   
            self.readout.add(layers.ReLU())
            self.readout.add(layers.Dropout(self.hparams['dropout_rate']))
        self.final = layers.Dense(
            self.output_units, 
            activation = self.final_activation,
            kernel_regularizer = tf.keras.regularizers.L2(self.hparams['l2_2']),
        )
        self.readout.build(input_shape = [None, self.hparams['path_state_dim']])
        self.final.build(input_shape = [None, self.hparams['path_state_dim']+readout_units])
        
    def _process_node_embeddings(self, inputs, relations, ti):
        if self.share:
            return self.node_update(inputs)
        else:
            return self.node_update[str(ti)](inputs)
    
    def init_path_state(self, bs, n_paths, f_path_init):
        # reshape for batches
        if bs > 0:
            f_path_init = tf.reshape(
                tf.transpose(
                    f_path_init,[1,0,2]
                ),[2,-1]
            )
        else:
            f_path_init = f_path_init
            
        #state init
        shape = tf.stack([n_paths, self.hparams['path_state_dim']-2], axis=0)
        path_state = tf.concat([
            tf.reshape(f_path_init[0],[-1,1]),
            tf.reshape(f_path_init[1],[-1,1]),
            tf.zeros(shape)
        ], axis=1)
        return path_state
    
    def call_embed(self, inputs, training=False):
        #call for batches
        f_ = inputs
        bs = len(f_['n_links']) if self.batch_size else self.batch_size
        n_paths = tf.math.reduce_sum(f_['n_paths'])
        n_links = tf.math.reduce_sum(f_['n_links'])
        n_nodes = tf.math.reduce_sum(f_['n_nodes'])
                
        #state init
        path_state = self.init_path_state(bs, n_paths, f_['path_init']) #(batch_size * n_flows, path_state_dim)
        
        shape = tf.stack([n_links, self.hparams['link_state_dim']-1], axis=0)
        temp = f_['link_init']
        link_state = tf.concat([
            tf.reshape(temp[temp>=0],[-1,1]),
            tf.zeros(shape)
        ], axis=1)

        shape = tf.stack([n_nodes,self.hparams['node_state_dim']-1], axis=0)
        node_state = tf.concat([
            tf.reshape(f_['node_init'],[-1,1]),
            tf.zeros(shape)
        ], axis=1)
        
        # pull for both
        temp = f_['sequences_paths_links']
        seqs = tf.reshape(temp[temp >= 0], [-1])
        if bs == 0:
            paths = f_['paths_to_links']
            links_to_paths = f_['links_to_paths']
            nodes_to_links = f_['nodes_to_links']
            nodes_to_paths = f_['nodes_to_paths']
            links_to_nodes = f_['links_to_nodes']
            support_matrix = f_[self.support]
        else:
            #pad path numbers
            paths_list = [] 
            for i in range(bs):
                temp = f_['paths_to_links'][i]
                paths_list.append(temp[temp>=0]+tf.math.reduce_sum(f_['n_paths'][:i]))
            paths = tf.concat(paths_list, axis=0)
            
            #pad link numbers
            links_list = []
            for i in range(bs):
                temp = f_['links_to_paths'][i]
                links_list.append(temp[temp>=0]+tf.math.reduce_sum(f_['n_links'][:i]))
            links_to_paths = tf.concat(links_list, axis=0)
            
            #pad link numbers
            nodes_to_links_list = []
            for i in range(bs):
                temp = f_['nodes_to_links'][i]
                nodes_to_links_list.append(temp[temp>=0] + tf.math.reduce_sum(f_['n_links'][:i]))
            nodes_to_links = tf.concat(nodes_to_links_list, axis=0)
            
            #pad node numbers
            nodes_to_paths_list = []
            for i in range(bs):
                temp = f_['nodes_to_paths'][i]
                nodes_to_paths_list.append(temp[temp>=0] + tf.math.reduce_sum(f_['n_nodes'][:i]))
            nodes_to_paths = tf.concat(nodes_to_paths_list, axis=0)
            
            #pad node numbers
            links_to_nodes_list = []
            for i in range(bs):
                temp = f_['links_to_nodes'][i]
                links_to_nodes_list.append(temp[temp>=0] + tf.math.reduce_sum(f_['n_nodes'][:i]))
            links_to_nodes = tf.concat(links_to_nodes_list, axis=0)
            
            #put together lap mats
            linop_blocks = [tf.linalg.LinearOperatorFullMatrix(block) for block in f_[self.support]]
            support_matrix = tf.linalg.LinearOperatorBlockDiag(linop_blocks)

        # iters
        for ti in range(self.hparams['T']):            
            ###################### PATH STATE #################################
            ids=tf.stack([paths, seqs], axis=tf.rank(paths)) #axis=1
            max_len = tf.reduce_max(seqs)+1
            lens = tf.math.segment_sum(data=tf.ones_like(paths), segment_ids=paths)
            
            # Collect link states of all the links included in all the paths 
            h_ = tf.gather(link_state, links_to_paths)
            shape = tf.stack([n_paths, max_len, self.hparams['link_state_dim']])
            link_inputs = tf.scatter_nd(ids, h_, shape)
            
            # Collect node states of all the nodes included in all the paths 
            h1_ = tf.gather(node_state, nodes_to_paths)
            shape = tf.stack([n_paths, max_len, self.hparams['node_state_dim']])
            node_inputs = tf.scatter_nd(ids, h1_, shape)
            
            # Concatenate link state with corresponding source node's state
            x_inputs = tf.concat([link_inputs, node_inputs], axis=tf.rank(link_inputs)-1) #axis=2
            
            # Update path state
            path_update = self.path_update if self.share else self.path_update[str(ti)]
            int_output, path_state = path_update(
                inputs        = x_inputs, 
                initial_state = path_state
            )
            
            if ti < self.hparams['T']-1:
                ###################### LINK STATE ################################# 
                m = tf.gather_nd(int_output,ids)
                m = tf.math.unsorted_segment_sum(
                    data         = m, 
                    segment_ids  = links_to_paths,
                    num_segments = n_links
                )
                #fitting nodes to links
                h2_ = tf.gather(node_state, links_to_nodes)
                _con = tf.concat([h2_, link_state, m], axis=1)
                link_state = self.edge_update(_con) if self.share else self.edge_update[str(ti)](_con)

                ###################### NODE STATE ################################# 
                node_state = self._process_node_embeddings(
                    (node_state, support_matrix, link_state),
                    (nodes_to_links, links_to_nodes, n_nodes),
                    ti if not self.share else None
                )        
        return path_state
       
    
    def call(self, inputs, training=False):     
        path_state = self.call_embed(inputs)
        if not self.hparams['learn_embedding']:
            path_state = tf.stop_gradient(path_state)
        r = self.readout(path_state,training=training)
        o = self.final(tf.concat([r,path_state], axis=1))
        return o
       

    def _get_labels_helper(self, labels):
        """
            When batched, single-category labels' shape: (BS, NUM_PATH), 
            which needs to be converted to (BS x NUM_PATH, ), and then
            to (BS x NUM_PATH, NUM_LABELS)
        """
        return tf.concat(
            [tf.reshape(
                tf.expand_dims(labels[t],1), 
                [-1,1]
            ) for t in self.train_on], 
            axis=1
        )              
    
    def train_step(self, data):
        features, labels = data 
        labels_on = self._get_labels_helper(labels)
        with tf.GradientTape() as tape:
            predictions = self(features, training=True)
            #print('train_step | pred:', tf.math.reduce_any(tf.math.is_nan(predictions)), predictions.shape)
            kpi_pred = predictions
            loss = self.loss_func(labels_on, kpi_pred) #tf.keras.metrics.mean_squared_error
            #print('train_step | loss:', tf.math.reduce_any(tf.math.is_nan(loss)), tf.math.reduce_mean(loss))

            regularization_loss = tf.math.reduce_sum(self.losses)
            total_loss = tf.math.reduce_sum(loss) + regularization_loss
            
        gradients = tape.gradient(total_loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))

        ret = {
            'loss':tf.math.reduce_sum(loss),
            'reg_loss':regularization_loss
        }
        for i,t in enumerate(self.train_on):
            ret[f'loss/{t}'] = loss[i]
            ret[f'label/mean/{t}'] = tf.math.reduce_mean(labels_on[:,i])
            ret[f'prediction/mean/{t}'] = tf.math.reduce_mean(kpi_pred[:,i])
        return ret
    

    def test_step(self, data):
        features, labels = data
        labels_on = self._get_labels_helper(labels)
        with tf.GradientTape() as tape:
            predictions = self(features, training=False)
            kpi_pred = predictions
            loss = self.loss_func(labels_on, kpi_pred) 
            regularization_loss = tf.math.reduce_sum(self.losses)
            maerr =tf.math.reduce_mean(
                tf.abs(labels_on - kpi_pred), 
                axis=0
            )
        ret = {
            'loss': tf.math.reduce_sum(loss),
            'mae': tf.math.reduce_sum(maerr),
            'reg_loss':regularization_loss
        }
        for i,t in enumerate(self.train_on):
            ret[f'loss/{t}'] = loss[i]
            ret[f'mae/{t}'] = maerr[i]
            ret[f'label/mean/{t}'] = tf.math.reduce_mean(labels_on[:,i])
            ret[f'prediction/mean/{t}'] = tf.math.reduce_mean(kpi_pred[:,i])
        return ret  
    
    
class PlanNetEmb(PlanNet): 
    def build_readout_layers(self):
        #readout-final
        self.readout, self.final = {}, {}
        if 'readoutLayerSizes' not in self.hparams:
            self.hparams['readoutLayerSizes'] = [self.hparams['readout_units']] * self.hparams['readout_layers']
        for lab in self.train_on:
            self.readout[lab] = tf.keras.models.Sequential(name=f'readout_{lab}') 
            for readout_units in self.hparams['readoutLayerSizes']:
                self.readout[lab].add(
                    layers.Dense(
                        readout_units, 
                        activation = None,
                        kernel_regularizer = tf.keras.regularizers.L2(self.hparams['l2'])
                    )
                )
                if self.use_norm:
                    self.readout[lab].add(layers.LayerNormalization()) 
                self.readout[lab].add(layers.ReLU())
                self.readout[lab].add(
                    layers.Dropout(
                        rate = self.hparams['dropout_rate']
                    )
                )
            self.final[lab] = layers.Dense(
                1, 
                kernel_regularizer = tf.keras.regularizers.L2(self.hparams['l2_2']),
                activation         = self.final_activation 
            )        
            
            self.readout[lab].build(input_shape = [None, self.hparams['path_state_dim']])
            self.final[lab].build(input_shape = [None, self.hparams['path_state_dim']+readout_units])
        
    def call(self, inputs, training=False):     
        path_state = self.call_embed(inputs, training)
        if not self.hparams['learn_embedding']:
            path_state = tf.stop_gradient(path_state)
        o = []  #readout
        for lab in self.train_on:
            r = self.readout[lab](path_state,training=training)
            o.append(self.final[lab](tf.concat([r,path_state], axis=1)))
        o = tf.concat(o, axis=1)
        return o    


class PlanNetGCN(PlanNet): 
    def _build_node_layers_helper(self, update_in, ti=None):  
        self.support = 'laplacian_matrix'
        node_update = GCNConvBN(
            self.hparams['node_state_dim'],
            batch_norm = self.use_norm, 
            dropout    = 0, #self.hparams['dropout_rate'],
            activation = tf.nn.relu,
            name = 'node_update' + (f'_t{ti}' if ti is not None else '')
        )        
        node_update.build(
            input_shape = [None, [update_in+self.hparams['link_state_dim']]]
        )
        return node_update

    def _process_node_embeddings(self, inputs, relations, ti):
        node_state, support_matrix, link_state = inputs
        nodes_to_links, links_to_nodes, n_nodes = relations
        h3_ = tf.gather(link_state, nodes_to_links)
        agg = tf.math.unsorted_segment_sum(
            data         = h3_, 
            segment_ids  = links_to_nodes,
            num_segments = n_nodes
        )
        _con2 = tf.concat([node_state, agg], axis=1)
        if self.share:
            return self.node_update((_con2, support_matrix))
        else:
            return self.node_update[str(ti)]((_con2, support_matrix))
    
    
class PlanNetEmbGCN(PlanNetEmb): 
    def _build_node_layers_helper(self, update_in, ti=None):  
        self.support = 'laplacian_matrix'
        node_update = GCNConvBN(
            self.hparams['node_state_dim'],
            batch_norm = self.use_norm, 
            dropout    = 0, #self.hparams['dropout_rate'],
            activation = tf.nn.relu,
            name = 'node_update' + (f'_t{ti}' if ti is not None else '')
        )        
        node_update.build(
            input_shape = [None, [update_in+self.hparams['link_state_dim']]]
        )
        return node_update
            
    def _process_node_embeddings(self, inputs, relations, ti):
        node_state, support_matrix, link_state = inputs
        nodes_to_links, links_to_nodes, n_nodes = relations
        h3_ = tf.gather(link_state, nodes_to_links)
        agg = tf.math.unsorted_segment_sum(
            data         = h3_, 
            segment_ids  = links_to_nodes,
            num_segments = n_nodes
        )
        _con2 = tf.concat([node_state, agg], axis=1)
        if self.share:
            return self.node_update((_con2, support_matrix)) 
        else:
            return self.node_update[str(ti)]((_con2, support_matrix)) 
    
    


    
# =====================
                
class RouteNetEmb(PlanNetEmb): 
    def __init__(self, hparams, output_units=1, batch_norm = False, batching=False, sharing=True ,
                 final_activation=None, train_on=['delay'], loss='mse', **argw):
        super().__init__(
            hparams, output_units, batch_norm, batching, sharing, final_activation, train_on, loss
        )
        self.hparams['node_state_dim'] = 0   
        
    def call_embed(self, inputs, training=False):
        #call for batches
        f_ = inputs
        bs = len(f_['n_links']) if self.batch_size else self.batch_size
        n_links = tf.math.reduce_sum(f_['n_links'])
        n_paths = tf.math.reduce_sum(f_['n_paths'])
                
        #state init
        path_state = self.init_path_state(bs, n_paths, f_['path_init']) #(batch_size * n_flows, path_state_dim)
        
        shape = tf.stack([n_links, self.hparams['link_state_dim']-1], axis=0)
        temp = f_['link_init']
        link_state = tf.concat([
            tf.reshape(temp[temp>=0],[-1,1]),
            tf.zeros(shape)
        ], axis=1)
        
        # pull for both
        temp = f_['sequences_paths_links']
        seqs = tf.reshape(temp[temp >= 0], [-1])
        if bs == 0:
            paths = f_['paths_to_links']
            links_to_paths = f_['links_to_paths']
        else:
            #pad path numbers
            paths_list = [] 
            for i in range(bs):
                temp = f_['paths_to_links'][i]
                paths_list.append(temp[temp>=0]+tf.math.reduce_sum(f_['n_paths'][:i]))
            paths = tf.concat(paths_list, axis=0)
            
            #pad link numbers
            links_list = []
            for i in range(bs):
                temp = f_['links_to_paths'][i]
                links_list.append(temp[temp>=0]+tf.math.reduce_sum(f_['n_links'][:i]))
            links_to_paths = tf.concat(links_list, axis=0)
            
        for ti in range(self.hparams['T']):
            ###################### PATH STATE #################################
            ids=tf.stack([paths, seqs], axis=tf.rank(paths)) #axis=1
            max_len = tf.reduce_max(seqs)+1
            lens = tf.math.segment_sum(data=tf.ones_like(paths), segment_ids=paths)
            
            # Collect link states of all the links included in all the paths 
            h_ = tf.gather(link_state, links_to_paths)
            shape = tf.stack([n_paths, max_len, self.hparams['link_state_dim']])
            link_inputs = tf.scatter_nd(ids, h_, shape)
            
            # Update path state
            path_update = self.path_update if self.share else self.path_update[str(ti)]
            int_output, path_state = path_update(
                inputs        = link_inputs, 
                initial_state = path_state
            )
            ###################### LINK STATE ################################# 
            if ti < self.hparams['T']-1:
                m = tf.gather_nd(int_output,ids)
                m = tf.math.unsorted_segment_sum(
                    data         = m, 
                    segment_ids  = links_to_paths,
                    num_segments = n_links
                )
                _con = tf.concat([link_state, m], axis=1)

                edge_update = self.edge_update if self.share else self.edge_update[str(ti)]
                link_state = edge_update(_con)

        return path_state        
        
        
###########################################
########## legacy: pending update #########   
###########################################
class RouteNet(PlanNet):  
    def __init__(self, hparams, output_units=1, batch_norm =True, batching=True, sharing=True, 
                 final_activation=None, train_on=['delay'], loss='mse'):
        super().__init__(
            hparams, output_units, batch_norm, batching, sharing, final_activation, train_on, loss
        )
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
            r = self.readout(path_state)#, training=training)
            o = self.final(tf.concat([r, path_state], axis=1))
        else:
            r = self.readout(tf.stop_gradient(path_state))#, training=training)
            o = self.final(tf.concat([r, tf.stop_gradient(path_state)], axis=1) )

        return o
    
    
    
class GenGNN(PlanNet):   
    def __init__(self, hparams, output_units=1, batch_norm = False, batching=False, sharing=True ,
                 final_activation=None, train_on=['delay'], loss='mse', **argw):
        super().__init__(
            hparams, output_units, batch_norm, batching, sharing, final_activation, train_on, loss
        )

    def _build_dims_helper(self):
        pass
        
    def build(self):
        self.conv0 = GeneralConv(channels = 64, n_layers = 2, activation="relu", dropout=dp_rate)
        self.conv1 = GeneralConv(channels = 128, n_layers = 2, activation="relu", dropout=dp_rate)
        self.conv2 = GeneralConv(channels = 64, n_layers = 2, activation="relu", dropout=dp_rate)
        self.global_pool = GlobalSumPool() 
        self.dense = Dense(self.output_units)

    def call(self, inputs):
        x, a, i = inputs
        x = self.conv0([x, a])
        x = self.conv1([x, a])
        x = self.conv2([x, a])
        x_batch = tf.reshape(x, (tf.reduce_max(i)+1, -1, x.shape[-1]))
        output = self.global_pool(x_batch)
        output = self.dense(output)
        
        return output
    