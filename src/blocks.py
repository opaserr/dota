# -*- coding: utf-8 -*-
# Building blocks of the data-driven dose calculator.
# COPYRIGHT: TU Delft, Netherlands. 2021.
import numpy as np
import tensorflow as tf
from tensorflow.keras import Sequential, layers
from tensorflow_addons.layers import GroupNormalization

class ConvEncoder(layers.Layer):
    """ Learnable position embedding + initial projection."""
    def __init__(self, num_tokens, projection_dim, num_channels=64,
                kernel_size=5, strides=1, pooling=2):
        super(ConvEncoder, self).__init__()
        self.num_tokens = num_tokens
        self.projection_dim = np.prod(projection_dim)
        self.projection_channels = projection_dim[-1]
        self.num_channels = num_channels
        self.kernel_size = kernel_size
        self.strides = strides
        self.pooling = pooling
        
        # Convolutional layers
        self.conv1 = layers.TimeDistributed(
            layers.Conv2D(self.num_channels,
                          (self.kernel_size,self.kernel_size),
                          strides=(self.strides,self.strides),
                          kernel_regularizer=ws_reg,
                          use_bias=False,
                          padding='same'))
        self.conv2 = layers.TimeDistributed(
            layers.Conv2D(self.num_channels, 
                          (self.kernel_size,self.kernel_size),
                          strides=(self.strides,self.strides),
                          kernel_regularizer=ws_reg,
                          use_bias=False,
                          padding='same'))
        
        # Final convolution
        self.conv3 = layers.TimeDistributed(
            layers.Conv2D(self.projection_channels, 
                          (self.kernel_size,self.kernel_size),
                          strides=(self.strides,self.strides),
                          kernel_regularizer=ws_reg,
                          use_bias=False,
                          padding='same'))
        
        # Pooling layers
        self.pool1 = layers.TimeDistributed(
            layers.MaxPooling2D(strides=(self.pooling,self.pooling)))
        self.pool2 = layers.TimeDistributed(
            layers.MaxPooling2D(strides=(self.pooling,self.pooling)))
        
        # Normalization layers
        self.norm1 = layers.TimeDistributed(
            GroupNormalization(groups=16))
        self.norm2 = layers.TimeDistributed(
            GroupNormalization(groups=16))
        self.norm3 = layers.TimeDistributed(
            GroupNormalization(groups=4))

        # Activation layers
        self.h1 = layers.TimeDistributed(layers.ReLU())
        self.h2 = layers.TimeDistributed(layers.ReLU())
        self.h3 = layers.TimeDistributed(layers.ReLU())
        
        # Flatten 3D slices to 1D vectors
        self.flatten = layers.TimeDistributed(
            layers.Flatten('channels_last'))
        
        # Positional encoding
        self.position_embedding = layers.Embedding(
            input_dim=self.num_tokens,
            output_dim=self.projection_dim)

        # Linear transformation of the energy
        self.linear = layers.Dense(units=self.projection_dim)
        self.reshape = layers.Reshape((1, self.projection_dim))
        self.concat = layers.Concatenate(axis=1)
        
    def call(self, tokens, energy_token):
        # First convolutional block
        x = self.h1(self.pool1(self.norm1(self.conv1(tokens))))
        
        # Second convolutional block
        x = self.h2(self.pool2(self.norm2(self.conv2(x))))
        
        # Last convolution + flatten to 1D vector
        x = self.flatten(self.h3(self.norm3(self.conv3(x))))

        # Encode energy and concatenate
        e = self.reshape(self.linear(energy_token))
        x = self.concat([e, x])

        # Add positional encoding
        positions = tf.range(start=0, limit=self.num_tokens, delta=1)
        encoded = x + self.position_embedding(positions)
        return encoded
    
class ConvDecoder(layers.Layer):
    """Convert transformer output to dose."""
    def __init__(self, projection_dim, num_channels=64,
                kernel_size=5, strides=2):
        super(ConvDecoder, self).__init__()
        self.projection_dim = projection_dim
        self.num_channels = num_channels
        self.kernel_size = kernel_size
        self.strides = strides
        
        # Reshape to 3D
        self.reshape = layers.TimeDistributed(
            layers.Reshape((projection_dim)))
        
        # Convolutional transpose layers
        self.conv1 = layers.TimeDistributed(
            layers.Conv2DTranspose(self.num_channels,
                                   (self.kernel_size,self.kernel_size),
                                   strides=(self.strides,self.strides),
                                   kernel_regularizer=ws_reg,
                                   use_bias=False,
                                   padding='same'))
        self.conv2 = layers.TimeDistributed(
            layers.Conv2DTranspose(self.num_channels,
                                   (self.kernel_size,self.kernel_size),
                                   strides=(self.strides,self.strides),
                                   kernel_regularizer=ws_reg,
                                   use_bias=False,
                                   padding='same'))
        
        # Output convolution
        self.conv3 = layers.TimeDistributed(
            layers.Conv2D(1, (self.kernel_size,self.kernel_size), padding='same'))
        
        # Normalization layers
        self.norm1 = layers.TimeDistributed(
            GroupNormalization(groups=16))
        self.norm2 = layers.TimeDistributed(
            GroupNormalization(groups=16))

        # Activation layers
        self.h1 = layers.TimeDistributed(layers.ReLU())
        self.h2 = layers.TimeDistributed(layers.ReLU())

        # Lambda layer to remove the first token
        self.slice = tf.keras.layers.Lambda(lambda x: x[:,1:,:,:])

    def call(self, tokens):
        # Reshape
        x = self.slice(self.reshape(tokens))
        
        # First convolutional block
        x = self.h1(self.norm1(self.conv1(x)))
        
        # Second convolutional block
        x = self.h2(self.norm2(self.conv2(x)))
        
        return self.conv3(x)

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

# Auxiliary functions
def ws_reg(kernel):
    """ Function for weight standardization"."""
    kernel_mean = tf.math.reduce_mean(kernel, axis=[0, 1, 2], keepdims=True, name='kernel_mean')
    kernel = kernel - kernel_mean
    kernel_std = tf.keras.backend.std(kernel, axis=[0, 1, 2], keepdims=True)
    kernel = kernel / (kernel_std + 1e-5)
    #return kernel