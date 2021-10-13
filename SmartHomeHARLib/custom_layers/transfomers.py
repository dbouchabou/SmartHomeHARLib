import numpy as np
from tqdm import tqdm
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

def create_padding_mask(seq):

    att_mask = []
    seq_length = seq.shape[1]
    with tqdm(total=seq.shape[0]+1, desc='Create padding mask') as pbar:
        masks = tf.cast(tf.math.equal(seq, 0), tf.float32)
        pbar.update(1)

        masks = 1 - masks

        for m in masks:
            am = np.ones((seq_length,seq_length))

            m1 = np.expand_dims(m, axis=0)
            m2 = np.expand_dims(m, axis=1)


            am = am*m1
            am = am*m2
            att_mask.append(am)

            pbar.update(1)


    att_mask = np.array(att_mask)

    return att_mask  # (batch_size, seq_len, seq_len)

def create_look_ahead_mask(size):

    mask = 1 - tf.linalg.band_part(tf.ones((size, size)), -1, 0)

    return mask  # (seq_len, seq_len)

def get_angles(pos, i, d_model):
    angle_rates = 1 / np.power(10000, (2 * (i//2)) / np.float32(d_model))
    return pos * angle_rates

def positional_encoding(position, d_model):
    angle_rads = get_angles(np.arange(position)[:, np.newaxis],
                            np.arange(d_model)[np.newaxis, :],
                            d_model)

    # apply sin to even indices in the array; 2i
    angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])

    # apply cos to odd indices in the array; 2i+1
    angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])

    pos_encoding = angle_rads[np.newaxis, ...]

    return tf.cast(pos_encoding, dtype=tf.float32)

class TokenAndPositionEmbedding(layers.Layer):
    def __init__(self, maxlen, vocab_size, embed_dim, **kwargs):
        super(TokenAndPositionEmbedding, self).__init__()

        self.vocab_size = vocab_size
        self.maxlen = maxlen
        self.embed_dim = embed_dim

        self.token_emb = layers.Embedding(input_dim=self.vocab_size, output_dim=self.embed_dim)
        #self.pos_emb = layers.Embedding(input_dim=self.maxlen, output_dim=self.embed_dim)
        self.pos_emb = positional_encoding(self.maxlen,
                                            self.embed_dim)

    def call(self, x):
        
        #positions = tf.range(start=0, limit=self.maxlen, delta=1)
        #positions = self.pos_emb(positions)
        #x = self.token_emb(x)
        #return x + positions
        x = self.token_emb(x)

        x = tf.keras.layers.Lambda(lambda x: tf.math.multiply(x,tf.math.sqrt(tf.cast(self.embed_dim, tf.float32))))(x)
        #x *= tf.math.sqrt(tf.cast(self.embed_dim, tf.float32))
        x = tf.keras.layers.Lambda(lambda x: tf.math.add(x,self.pos_emb[:, :self.maxlen, :]))(x)
        #x += self.pos_emb[:, :self.maxlen, :]

        return x


    def get_config(self):
        config = super().get_config()
        config.update({
            'token_emb': self.token_emb,
            'pos_emb': self.pos_emb,
            'vocab_size': self.vocab_size,
            'maxlen': self.maxlen,
            'embed_dim': self.embed_dim,
        })
        return config


class TransformerBlock(layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1, activation="relu", **kwargs):
        super(TransformerBlock, self).__init__()

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.ff_dim = ff_dim
        self.rate = rate
        self.activation = activation


        self.att = layers.MultiHeadAttention(num_heads=self.num_heads, key_dim=self.embed_dim)
        self.ffn = keras.Sequential(
            [layers.Dense(ff_dim, activation=self.activation), layers.Dense(self.embed_dim),]
        )
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(self.rate)
        self.dropout2 = layers.Dropout(self.rate)

    def call(self, inputs, training, mask=None):
        attn_output = self.att(inputs, inputs, attention_mask=mask)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)

    def get_config(self):
        config = super().get_config()
        config.update({
            'att': self.att,
            'ffn': self.ffn,
            'layernorm1': self.layernorm1,
            'layernorm2': self.layernorm2,
            'dropout1': self.dropout1,
            'dropout2': self.dropout2,
            'embed_dim': self.embed_dim,
            'num_heads': self.num_heads,
            'ff_dim': self.ff_dim,
            'rate': self.rate,
            'activation': self.activation
        })
        return config