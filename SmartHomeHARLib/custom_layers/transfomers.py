import numpy as np
from tqdm import tqdm
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

def padding_attention_mask(seq):

    att_mask = []

    seq_masks = tf.cast(tf.math.not_equal(seq, 0), tf.float32)

    for seq_mask in seq_masks:

        basic_att_mask = tf.ones((seq_mask.shape[1],seq_mask.shape[1]))

        # gerenrate de attention mask
        att_seq_mask = basic_att_mask*seq_mask*tf.transpose(seq_mask)

        # add atention mask of the sequence
        att_mask.append(att_seq_mask)

        # convert the mask to boolean
        att_mask = tf.cast(att_mask, bool)

    return att_mask  # (batch_size, seq_len, seq_len)

def causal_attention_mask(batch_size, n_dest, n_src, dtype):
    """
    Mask the upper half of the dot product matrix in self attention.
    This prevents flow of information from future tokens to current token.
    1's in the lower triangle, counting from the lower right corner.
    """
    i = tf.range(n_dest)[:, None]
    j = tf.range(n_src)
    m = i >= j - n_src + n_dest
    mask = tf.cast(m, dtype)
    mask = tf.reshape(mask, [1, n_dest, n_src])
    mult = tf.concat(
        [tf.expand_dims(batch_size, -1), tf.constant([1, 1], dtype=tf.int32)], 0
    )
    return tf.tile(mask, mult)

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

    def call(self, inputs, mask=None):
        attn_output = self.att(inputs, inputs, attention_mask=mask)
        attn_output = self.dropout1(attn_output)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output)
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


class GPTDecoder(layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1, activation="gelu", **kwargs):
        super(GPTDecoder, self).__init__()

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.ff_dim = ff_dim
        self.rate = rate
        self.activation = activation

        self.block = TransformerBlock(self.embed_dim, self.num_heads, self.ff_dim, self.rate, self.activation)

       # self.att = layers.MultiHeadAttention(self.num_heads, self.embed_dim)
        #self.ffn = keras.Sequential(
        #    [layers.Dense(self.ff_dim, activation=self.activation), layers.Dense(self.embed_dim),]
        #)
        #self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        #self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        #self.dropout1 = layers.Dropout(self.rate)
        #self.dropout2 = layers.Dropout(self.rate)

    def call(self, inputs):
        input_shape = tf.shape(inputs)
        batch_size = input_shape[0]
        seq_len = input_shape[1]
        
        causal_mask = causal_attention_mask(batch_size, seq_len, seq_len, tf.bool)
        
        out = self.block(inputs,causal_mask)

        #attention_output = self.att(inputs, inputs, attention_mask=causal_mask)
        #attention_output = self.dropout1(attention_output)
        #out1 = self.layernorm1(inputs + attention_output)
        #ffn_output = self.ffn(out1)
        #ffn_output = self.dropout2(ffn_output)
        #return self.layernorm2(out1 + ffn_output)

        return out
    
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