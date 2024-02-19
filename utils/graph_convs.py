import warnings
import tensorflow as tf
#print(tf.test.gpu_device_name())

##
from tensorflow.keras import backend as K
from tensorflow.keras.layers import LayerNormalization, Dropout, Input, Dense, Lambda, Layer
from tensorflow.keras.initializers import Constant
from tensorflow.keras.models import Model
from spektral.layers import ops, ECCConv
from spektral.layers.convolutional.conv import Conv
from spektral.utils import gcn_filter
from spektral.layers.ops import modes


class GCNConvBN(Conv):
    # https://github.com/danielegrattarola/spektral/blob/master/spektral/layers/convolutional/general_conv.py
    def __init__(
        self,
        channels,
        batch_norm=True,
        dropout=0.0,
        activation=None,
        use_bias=True,
        kernel_initializer="glorot_uniform",
        bias_initializer="zeros",
        kernel_regularizer=None,
        bias_regularizer=None,
        activity_regularizer=None,
        kernel_constraint=None,
        bias_constraint=None,
        **kwargs,
    ):
        super().__init__(
            activation=activation,
            use_bias=use_bias,
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer,
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer,
            activity_regularizer=activity_regularizer,
            kernel_constraint=kernel_constraint,
            bias_constraint=bias_constraint,
            **kwargs,
        )
        self.channels = channels
        self.use_norm = batch_norm
        self.dropout_rate = dropout

    def build(self, input_shape):
        assert len(input_shape) >= 2
        input_dim = input_shape[1][-1]
        self.dropout = Dropout(self.dropout_rate)
        if self.use_norm:
            self.batch_norm = LayerNormalization()
        self.kernel = self.add_weight(
            shape=(input_dim, self.channels),
            initializer=self.kernel_initializer,
            name="kernel",
            regularizer=self.kernel_regularizer,
            constraint=self.kernel_constraint,
        )
        if self.use_bias:
            self.bias = self.add_weight(
                shape=(self.channels,),
                initializer=self.bias_initializer,
                name="bias",
                regularizer=self.bias_regularizer,
                constraint=self.bias_constraint,
            )
        self.built = True      
        
    def call(self, inputs, mask=None):
        x, a = inputs
        
        output = K.dot(x, self.kernel)
        output = ops.modal_dot(a, output)
        
        if self.use_bias:
            output = K.bias_add(output, self.bias)
            
        # with batch norm and dropout
        if self.use_norm:
            output = self.batch_norm(output)
        output = self.dropout(output)
        
        if mask is not None:
            output *= mask[0]
        output = self.activation(output)

        return output
    
    @property
    def config(self):
        return {"channels": self.channels}

    @staticmethod
    def preprocess(a):
        return gcn_filter(a)
    
    


class ECCConvBN(Conv):
    # https://github.com/danielegrattarola/spektral/blob/master/spektral/layers/convolutional/ecc_conv.py
    def __init__(
        self,
        channels,
        kernel_network=None,
        root=True,
        batch_norm=True,
        dropout=0.0,        
        activation=None,
        use_bias=True,
        kernel_initializer="glorot_uniform",
        bias_initializer="zeros",
        kernel_regularizer=None,
        bias_regularizer=None,
        activity_regularizer=None,
        kernel_constraint=None,
        bias_constraint=None,
        **kwargs,
    ):
        super().__init__(
            activation=activation,
            use_bias=use_bias,
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer,
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer,
            activity_regularizer=activity_regularizer,
            kernel_constraint=kernel_constraint,
            bias_constraint=bias_constraint,
            **kwargs,
        )
        self.channels = channels
        self.kernel_network = kernel_network
        self.root = root
        self.use_norm = batch_norm
        self.dropout_rate = dropout
        
    def build(self, input_shape):
        self.dropout = Dropout(self.dropout_rate)
        if self.use_norm:
            self.batch_norm = LayerNormalization()
        
        F = input_shape[1][-1]
        F_ = self.channels   
        
        self.kernel_network_layers = []
        if self.kernel_network is not None:
            for i, l in enumerate(self.kernel_network):
                self.kernel_network_layers.append(
                    Dense(
                        l,
                        name="FGN_{}".format(i),
                        activation="relu",
                        use_bias=self.use_bias,
                        kernel_initializer=self.kernel_initializer,
                        bias_initializer=self.bias_initializer,
                        kernel_regularizer=self.kernel_regularizer,
                        bias_regularizer=self.bias_regularizer,
                        kernel_constraint=self.kernel_constraint,
                        bias_constraint=self.bias_constraint,
                        dtype=self.dtype,
                    )
                )
        self.kernel_network_layers.append(
            Dense(F_ * F, dtype=self.dtype, name="FGN_out")
        )
        for layer in self.kernel_network_layers:
            layer.build([None, F])

        if self.root:
            self.root_kernel = self.add_weight(
                name="root_kernel",
                shape=(F, F_),
                initializer=self.kernel_initializer,
                regularizer=self.kernel_regularizer,
                constraint=self.kernel_constraint,
            )
        if self.use_bias:
            self.bias = self.add_weight(
                name="bias",
                shape=(self.channels,),
                initializer=self.bias_initializer,
                regularizer=self.bias_regularizer,
                constraint=self.bias_constraint,
            )
        self.built = True

    def call(self, inputs, mask=None):
        x, a, e = inputs
        
        a = a.to_dense()
        a_bi = tf.cast(a > 0, tf.int32)

        # Parameters
        N = tf.shape(x)[-2]
        F = tf.shape(x)[-1]
        F_ = self.channels

        # Filter network
        kernel_network = e
        for layer in self.kernel_network_layers:
            kernel_network = layer(kernel_network)

        # Convolution
        mode = ops.autodetect_mode(x, a)  # mode={single, disjoint, batch, mixed}
        if mode == modes.BATCH:
            kernel = K.reshape(kernel_network, (-1, N, N, F_, F))
            output = kernel * a_bi[..., None, None]
            output = tf.einsum("abcde,ace->abd", output, x)
        else:
            # Enforce sparse representation
            if not K.is_sparse(a):
#                 warnings.warn(
#                     "Casting dense adjacency matrix to SparseTensor."
#                     "This can be an expensive operation. "
#                 )
                a = tf.sparse.from_dense(a)
            if not K.is_sparse(a_bi):
                a_bi = tf.sparse.from_dense(a_bi)

            target_shape = (-1, F, F_)
            if mode == modes.MIXED: 
                target_shape = (tf.shape(x)[0],) + target_shape
            
            # edge weights
            kernel_network = tf.reshape(a.values, [-1,1]) * kernel_network
            kernel = tf.reshape(kernel_network, target_shape)
            
            index_targets = a_bi.indices[:, 1]
            index_sources = a_bi.indices[:, 0]
            
            messages = tf.gather(x, index_sources, axis=-2)
            messages = tf.einsum("...ab,...abc->...ac", messages, kernel)
            output = ops.scatter_sum(messages, index_targets, N)

        if self.root:
            output += K.dot(x, self.root_kernel)
        if self.use_bias:
            output = K.bias_add(output, self.bias)
            
        # with batch norm and dropout
        if self.use_norm:
            output = self.batch_norm(output)
        output = self.dropout(output)
        
        if mask is not None:
            output *= mask[0]
        output = self.activation(output)

        return output

    @property
    def config(self):
        return {
            "channels": self.channels,
            "kernel_network": self.kernel_network,
            "root": self.root,
        }