import tensorflow as tf
import spektral as spk
import json

class unoNet(tf.keras.Model):
    """ 
      This class initialzes a model, trains it and evaluates
      on both training dataset and testing dataset(or validation).
      The loss for training and testing might be different,
      therefore it is split into two different methods.
      Overall, this class follows Functional API for TF2,
      and incldues classic __init__, build, and call methods.
    """
    def __init__(self,config, output_units=1):
        """
          Initializes the model using config file.
        """
        super(unoNet, self).__init__()

        self.config = config
        self.output_units = output_units
        
    def build(self, input_shape=None):
        """
          Builds the NN model and initializes
          all layers and parameters and hyperparameters of it.

        """
        del input_shape

        self.path_update = tf.keras.layers.RNN(
            tf.keras.layers.GRUCell(int(self.config['GNN']['path_state_dim'])),
            return_sequences=True,
            return_state=True,
            dtype=tf.float32)
        self.node_update = spk.layers.GCNConv(int(self.config['GNN']['node_state_dim']))
        

        ## To DO - Introduce layers with different sizes
        self.edge_update = tf.keras.models.Sequential(name="edge_update")
        edgeLayerSizes = json.loads(self.config.get('GNN','edgeMLPLayerSizes'))
        for i in range(len(edgeLayerSizes)):
            self.edge_update.add(tf.keras.layers.Dense(edgeLayerSizes[i],
                                activation=tf.nn.relu,
                                kernel_regularizer=tf.keras.regularizers.L2(float(self.config['LearningParams']['l2']))))
        self.edge_update.add(tf.keras.layers.Dense(int(self.config['GNN']['link_state_dim'])))
        
        
        #readout-final
        self.readout = tf.keras.models.Sequential(name='readout')

        for i in range(int(self.config['GNN']['readout_layers'])):
            self.readout.add(tf.keras.layers.Dense(int(self.config['GNN']['readout_units']), 
                    activation=tf.nn.relu,
                    kernel_regularizer=tf.keras.regularizers.L2(float(self.config['LearningParams']['l2']))))

            self.readout.add(tf.keras.layers.Dropout(rate=float(self.config['LearningParams']['dropout_rate'])))

        self.final = tf.keras.layers.Dense(self.output_units, 
                kernel_regularizer=tf.keras.regularizers.L2(float(self.config['LearningParams']['l2_2'])))

        self.edge_update.build(tf.TensorShape([None,int(self.config['GNN']['link_state_dim'])+
                                            int(self.config['GNN']['node_state_dim'])+
                                            int(self.config['GNN']['path_state_dim'])]))

        self.readout.build(input_shape = [None,int(self.config['GNN']['path_state_dim'])])
        self.final.build(input_shape = [None,int(self.config['GNN']['path_state_dim']) + int(self.config['GNN']['readout_units']) ])
        self.built = True

    def call(self, inputs, training=False):
        """
          Makes transformation from inputs to outputs and backwards.
          Essentially makes forward and backward passes.
          All updates happen here.

        """
        
        f_ = inputs
        # Link state initialization
        shape = tf.stack([f_["n_links"],int(self.config['GNN']['link_state_dim'])-1], axis=0)
        link_state = tf.concat([
            # tf.expand_dims(f_['link_init'],axis=1),
            tf.expand_dims(f_["capacities"],axis=1),
            tf.zeros(shape)
        ], axis=1)

        # Node state initialization
        shape = tf.stack([f_["n_nodes"],int(self.config['GNN']['node_state_dim'])-1], axis=0)
        node_state = tf.concat([
            tf.expand_dims(f_["degrees"],axis=1),
            tf.zeros(shape)
        ], axis=1)

        # Path state initialization
        shape = tf.stack([f_["n_paths"],int(self.config['GNN']['path_state_dim'])-2], axis=0)
        path_state = tf.concat([
            tf.expand_dims(f_["traffic"][0],axis=1),
            tf.expand_dims(f_["traffic"][1],axis=1),
            tf.zeros(shape)
        ], axis=1)
        
        for _ in range(int(self.config['GNN']['T'])):

            ###################### PATH STATE #################################

            ids=tf.stack([f_["paths_to_links"], f_["sequences_paths_links"]], axis=1)
            max_len = tf.reduce_max(f_["sequences_paths_links"])+1 # Length of the path with maximum number of links

            # Collect link states of all the links included in all the paths 
            h_ = tf.gather(link_state,f_["links_to_paths"])
            shape = tf.stack([f_["n_paths"], max_len, int(self.config['GNN']['link_state_dim'])])
            link_inputs = tf.scatter_nd(ids, h_, shape)

            # Collect node states of all the nodes included in all the paths 
            h1_ = tf.gather(node_state,f_["nodes_to_paths"])
            shape = tf.stack([f_["n_paths"], max_len, int(self.config['GNN']['node_state_dim'])])
            node_inputs = tf.scatter_nd(ids, h1_, shape)

            # Concatenate link state with corresponding source node's state
            x_inputs = tf.concat([link_inputs, node_inputs], axis=2)

            # Update path state
            outputs, path_state = self.path_update(inputs = x_inputs,
                                                initial_state = path_state)

            ###################### LINK STATE #################################                                  
            m = tf.gather_nd(outputs,ids)
            m = tf.math.unsorted_segment_sum(data=m, 
                                            segment_ids=f_["links_to_paths"],
                                            num_segments=f_["n_links"]) # TO DO:  Verify num_segments 
            # Collect node states of all the nodes included in all the links 
            h2_ = tf.gather(node_state,f_["links_to_nodes"])            
            _con = tf.concat([h2_, link_state, m], axis=1)
            link_state = self.edge_update(_con)

            ###################### NODE STATE ################################# 
            h3_ = tf.gather(link_state, f_["nodes_to_links"])
            agg = tf.math.unsorted_segment_sum(data=h3_, 
                                            segment_ids=f_["links_to_nodes"],
                                            num_segments=f_["n_nodes"])
            _con2 = tf.concat([node_state, agg], axis=1)
            node_state = self.node_update((_con2,f_["laplacian_matrix"]))
        # Readout        
        if self.config['LearningParams']['learn_embedding']:
            r = self.readout(path_state,training=training)
            o = self.final(tf.concat([r,path_state], axis=1)) 
        else:
            r = self.readout(tf.stop_gradient(path_state),training=training)
            o = self.final(tf.concat([r, tf.stop_gradient(path_state)], axis=1))
        return o

    @tf.function
    def train_step(self, data):
        """ 
          Method to perform one step of training on inputted data.
        """
        # print(data)
        features, labels = data
        
        with tf.GradientTape() as tape:
            predictions = self(features, training=True)
            print(tf.math.is_nan(predictions))
            loc  = predictions[...,0]
            delay_prediction = loc
            loss = tf.keras.metrics.mean_squared_error(labels[0], loc)
            print(tf.math.is_nan(loss))

            regularization_loss = sum(self.losses)
            total_loss = loss + regularization_loss
            
        gradients = tape.gradient(total_loss, self.trainable_variables)        
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
        ret = {
            'loss':loss,
            'label/mean/delay':tf.math.reduce_mean(labels[0]),
            'prediction/mean/delay': tf.math.reduce_mean(delay_prediction)
            }
        return ret

    @tf.function
    def test_step(self, data):
        """ 
          Method to perform one step of evaluating on inputted data.
        """
        features, labels = data
        
        with tf.GradientTape() as tape:
            predictions = self(features, training=False)
            loc  = predictions[...,0]
            delay_prediction = loc
            loss = tf.keras.metrics.mean_squared_error(labels[0], loc)
            regularization_loss = sum(self.losses)
            total_loss = loss + regularization_loss
            
        ret = {
            'loss':loss,
            'label/mean/delay':tf.math.reduce_mean(labels[0]),
            'prediction/mean/delay': tf.math.reduce_mean(delay_prediction)
            }
        return ret