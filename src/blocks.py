# -*- coding: utf-8 -*-
# Building blocks of the data-driven dose calculator.
# COPYRIGHT: TU Delft, Netherlands. 2021.
import numpy as np
import tensorflow as tf
from tensorflow.keras import Sequential, layers
from tensorflow_addons.layers import GroupNormalization
from tensorflow.keras.layers import TimeDistributed as td

class ConvEncoder(layers.Layer):
    """ Tokenize 2D using a series of convolutional layers."""
    def __init__(self, filters, steps=4, num_channels=32, kernel_size=5):
        super(ConvEncoder, self).__init__()
        self.projection_channels = filters
        self.num_channels = num_channels
        self.kernel_size = kernel_size
        self.down_steps = steps
        self.encoder = Sequential()
        
        # Build list of layers - conv, pooling, normalization, activation
        for _ in range(self.down_steps):
            self.encoder.add(ConvBlock(num_channels, kernel_size, downsample=True))

        # Final convolution and flatten stack
        self.encoder.add(ConvBlock(filters, kernel_size, flatten=True))
        
    def call(self, volumes):
        return self.encoder(volumes)

    
class ConvDecoder(layers.Layer):
    """Convert transformer output to dose."""
    def __init__(self, token_dim, steps=4, num_channels=32, kernel_size=5):
        super(ConvDecoder, self).__init__()
        self.token_dim = token_dim
        self.num_channels = num_channels
        self.kernel_size = kernel_size
        self.up_steps = steps
        self.decoder = Sequential()
        
        # Reshape to 3D
        self.reshape = td(layers.Reshape((token_dim)))
        
        # Convolutional transpose layers
        for _ in range(self.up_steps):
            self.decoder.add(ConvBlock(num_channels, kernel_size, upsample=True))
        self.decoder.add(td(layers.Conv2D(1, kernel_size, padding='same')))

    def call(self, tokens):
        return self.decoder(self.reshape(tokens))


class TransformerEncoder(layers.Layer):
    """ Transformer encoder block."""
    def __init__(self,  num_heads, num_tokens, projection_dim, causal=True,
                dropout_rate=0.2, num_forward=0):
        super(TransformerEncoder, self).__init__()
        self.num_tokens = num_tokens
        self.num_heads = num_heads
        self.projection_dim = np.prod(projection_dim)
        
        # Multi-head self attention layer
        self.multihead = layers.MultiHeadAttention(
            num_heads=self.num_heads,
            key_dim=self.projection_dim,
            dropout=dropout_rate,
            kernel_initializer='truncated_normal',
            use_bias=False)
        
        # MLP stack layer
        self.mlp_network = Sequential([
            layers.Dense(self.projection_dim, activation=tf.nn.gelu),
            layers.Dropout(dropout_rate),
            layers.Dense(self.projection_dim, activation=tf.nn.gelu),
            layers.Dropout(dropout_rate)
            ])
        
        # Normalization layers
        self.norm1 = layers.LayerNormalization(epsilon=1e-6)
        self.norm2 = layers.LayerNormalization(epsilon=1e-6)
        self.add = layers.Add()
        
        # Mask for causal attention
        if causal:
            self.mask = np.tri(num_tokens, num_tokens, num_forward, dtype=bool) 
        else:
            self.mask = np.ones((num_tokens, num_tokens))

    def call(self, tokens):
        x = self.norm1(tokens)
        x = self.multihead(x, x, attention_mask=self.mask)
        x = self.add([x, tokens])
        y = self.norm2(x)
        y = self.mlp_network(y)
        return self.add([x,y])


##################################
# Auxiliary blocks and functions
##################################

class ConvBlock(layers.Layer):
    """ Down-sampling convolutional block."""
    def __init__(self, num_channels=64, kernel_size=3, downsample=False,
        upsample=False, flatten=False):
        super(ConvBlock, self).__init__()
        
        # Build list of layers - conv, pooling, normalization, activation
        self.block = Sequential()
        self.block.add(layers.Conv3D(
            num_channels, kernel_size, kernel_regularizer=ws_reg, use_bias=False, padding='same'))
        self.block.add(layers.MaxPooling3D(pool_size=(1,2,2))) if downsample else None
        self.block.add(layers.UpSampling3D(size=(1,2,2))) if upsample else None
        self.block.add(layers.LayerNormalization())
        self.block.add(layers.LeakyReLU())
        self.block.add(td(layers.Flatten('channels_last'))) if flatten else None
        
    def call(self, inputs):
        return self.block(inputs)


class PosEmbedding(layers.Layer):
    """ Concatenate sequences and append position embedding."""
    def __init__(self, num_tokens, token_size):
        super(PosEmbedding, self).__init__()
        self.num_tokens = num_tokens
        self.token_size = token_size
        self.concat = layers.Concatenate(axis=1)
        self.embedding = layers.Embedding(
            input_dim=num_tokens, output_dim=token_size)

    def call(self, *args):
        positions = tf.range(start=0, limit=self.num_tokens, delta=1)
        return self.concat(list(args)) + self.embedding(positions)


class LinearProj(layers.Layer):
    """ Project scalars to token vectors."""
    def __init__(self, token_size):
        super(LinearProj, self).__init__()
        self.token_size = token_size
        self.projection = td(layers.Dense(token_size, use_bias=False))
        self.flatten = td(layers.Flatten('channels_last'))

    def call(self, inputs):
        return self.projection(self.flatten(inputs))


def ws_reg(kernel):
    """ Function for weight standardization"."""
    kernel_mean = tf.math.reduce_mean(kernel, axis=[0, 1, 2], keepdims=True, name='kernel_mean')
    kernel = kernel - kernel_mean
    kernel_std = tf.keras.backend.std(kernel, axis=[0, 1, 2], keepdims=True)
    kernel = kernel / (kernel_std + 1e-5)
    #return kernel