# -*- coding: utf-8 -*-
# Models of the data-driven dose calculator.
# COPYRIGHT: TU Delft, Netherlands. 2021.
import numpy as np
from tensorflow import concat as cat
from tensorflow.keras import Sequential, layers, Model
from blocks import ConvEncoder, ConvDecoder, ConvBlock, TransformerEncoder, PosEmbedding

def dota_photons(inshape, steps, enc_feats, num_heads, num_transformers,
    kernel_size, dropout_rate=0.2, causal=False):
    """ Creates the transformer model for dose calculation using multiple
    energies and patients."""

    # Calculate size of input tokens
    slice_dim = inshape[1:]
    token_dim = (*[int(i/2**steps) for i in slice_dim[:-1]], enc_feats)
    token_size = np.prod(token_dim)
    num_tokens = inshape[0]

    # Input CT and ray tracing values
    ct_vol = layers.Input((num_tokens, *slice_dim))
    ray_tr = layers.Input((num_tokens, *slice_dim))
    inputs = layers.Concatenate()([ct_vol, ray_tr])

    # Encode inputs + positional embedding
    tokens = ConvEncoder(enc_feats, steps=steps, kernel_size=kernel_size)(inputs)
    tokens = PosEmbedding(num_tokens, token_size)(tokens)
    
    # Stack transformer encoders
    for i in range(num_transformers):

        # Transformer encoder blocks
        tokens = TransformerEncoder(num_heads, num_tokens, token_size,
            causal=causal, dropout_rate=dropout_rate)(tokens)

    # Decode and upsample
    outputs = ConvDecoder(token_dim, steps=steps, kernel_size=kernel_size)(tokens)

    return Model(inputs=[ct_vol, ray_tr], outputs=outputs)


def dota_residual(inshape, steps, enc_feats, num_heads, num_transformers,
    kernel_size, dropout_rate=0.2, causal=True):
    """ Creates the transformer model for dose calculation using multiple
    energies and patients."""

    # Calculate size of input tokens
    slice_dim = inshape[1:]
    token_dim = (*[int(i/2**steps) for i in slice_dim[:-1]], enc_feats)
    token_size = np.prod(token_dim)
    num_tokens = inshape[0]

    # Input CT and ray tracing values
    ct_vol = layers.Input((num_tokens, *slice_dim))
    ray_tr = layers.Input((num_tokens, *slice_dim))
    x = layers.Concatenate()([ct_vol, ray_tr])
    x_history = [x]

    # Encode inputs
    for _ in range(steps):
        x = ConvBlock(kernel_size=kernel_size, downsample=True)(x)
        x_history.append(x)

    # Tokenize + positional embedding
    tokens = ConvBlock(enc_feats, kernel_size, flatten=True)(x)
    tokens = PosEmbedding(num_tokens, token_size)(tokens)
    
    # Stack transformer encoders
    for i in range(num_transformers):

        # Transformer encoder blocks
        tokens = TransformerEncoder(num_heads, num_tokens, token_size,
            causal=causal, dropout_rate=dropout_rate)(tokens)

    # Reshape to cube
    x = layers.TimeDistributed(layers.Reshape((token_dim)))(tokens)

    # Decode and upsample
    for _ in range(steps):
        x = cat([x, x_history.pop()], axis=-1)
        x = ConvBlock(kernel_size=kernel_size, upsample=True)(x)

    dose = layers.Conv3D(1, kernel_size, padding='same')(x)

    return Model(inputs=[ct_vol, ray_tr], outputs=dose)
