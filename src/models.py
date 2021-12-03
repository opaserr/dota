# -*- coding: utf-8 -*-
# Models of the data-driven dose calculator.
# COPYRIGHT: TU Delft, Netherlands. 2021.
import numpy as np
from tensorflow.keras import Sequential, layers, Model
from .blocks import ConvEncoder, ConvDecoder, TransformerEncoder

def multi_energy_model(num_tokens, input_shape, projection_dim,
    num_heads, num_transformers, kernel_size, dropout_rate=0.2,
    causal=True):
    """ Creates the transformer model for dose calculation using multiple
    energies and patients."""

    # Input CT values
    inputs = layers.Input((num_tokens-1, *input_shape))

    # Input energies
    energies = layers.Input((1))

    # Encode inputs + positional embedding
    tokens = ConvEncoder(
        num_tokens,
        projection_dim,
        kernel_size=kernel_size, 
    )(inputs, energies)
    
    # Stack transformer encoders
    for i in range(num_transformers):

        # Transformer encoder blocks.
        tokens = TransformerEncoder(
            num_heads, 
            num_tokens, 
            projection_dim,
            causal=causal,
            dropout_rate = dropout_rate,
        )(tokens)

    # Decode and upsample
    outputs = ConvDecoder(
        projection_dim, 
        kernel_size=kernel_size,
    )(tokens)

    return Model(inputs=[inputs, energies], outputs=outputs)
