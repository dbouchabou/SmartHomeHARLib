import numpy as np
from tqdm import tqdm
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import math


def create_padding_mask_gpt(seq):
    seq = tf.cast(tf.math.equal(seq, 0), tf.float32)
    # add extra dimensions to add the padding
    # to the attention logits.
    seq = tf.expand_dims(seq, axis=1)
    seq = tf.expand_dims(seq, axis=1)
    return seq  * -1e9

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
    print(att_mask.shape)

    return att_mask  # (batch_size, seq_len, seq_len)

@tf.function
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

def create_padding_mask_origine(seq):
  seq = tf.cast(tf.math.equal(seq, 0), tf.float32)

  # add extra dimensions to add the padding
  # to the attention logits.
  return seq[:, tf.newaxis, tf.newaxis, :]  # (batch_size, 1, 1, seq_len)

def create_padding_mask_2(seq, n_dest):
    seq = tf.cast(tf.math.equal(seq, 0), tf.float32)
    
    mask = tf.cast(seq, dtype)

    o = tf.ones((n_dest,n_src))

    seq_2 = seq*o

    
    return seq_2




def padding_attention_mask_2(seq):

    seq_mask = tf.cast(tf.math.not_equal(seq, 0), tf.float32)

    input_shape = tf.shape(seq_mask)
    batch_size = input_shape[0]
    seq_len = input_shape[1]

    basic_att_mask = tf.ones((batch_size,seq_len,seq_len))

    # gerenrate de attention mask
    #att_seq_mask = basic_att_mask*seq_mask*tf.transpose(seq_mask)
    #att_seq_mask = basic_att_mask*tf.expand_dims(seq_mask, 2)
    att_seq_mask = basic_att_mask*tf.expand_dims(seq_mask, 1)
    #att_seq_mask = basic_att_mask*tf.transpose(tf.expand_dims(seq_mask, 1), perm=[0, 2, 1])

    # convert the mask to boolean
    att_mask = tf.cast(att_seq_mask, bool)

    return att_mask  # (batch_size, seq_len, seq_len)

# celle qui marche
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

def padding_attention_mask_3(seq):
    seq = tf.cast(tf.math.not_equal(seq, 0), tf.float32)
    mask = seq[:, :, tf.newaxis]

    mask = mask*tf.transpose(mask, perm=[0, 2, 1])
    # add extra dimensions to add the padding
    # to the attention logits.
    return tf.cast(mask, tf.bool)


#def create_look_ahead_mask(size):
#
#    mask = 1 - tf.linalg.band_part(tf.ones((size, size)), -1, 0)
#
#    return mask  # (seq_len, seq_len)

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

        self.supports_masking = True
        self.vocab_size = vocab_size
        self.maxlen = maxlen
        self.embed_dim = embed_dim
        self.mask = None

        self.token_emb = layers.Embedding(input_dim=self.vocab_size, output_dim=self.embed_dim, mask_zero = True)
        self.pos_emb = layers.Embedding(input_dim=self.maxlen, output_dim=self.embed_dim, mask_zero = True)
        #self.token_emb = layers.Embedding(input_dim=self.vocab_size, output_dim=self.embed_dim, mask_zero = False)
        #self.pos_emb = layers.Embedding(input_dim=self.maxlen, output_dim=self.embed_dim, mask_zero = False)
        #self.pos_emb = positional_encoding(self.maxlen,self.embed_dim)

    def call(self, x):
        
        self.mask = self.token_emb.compute_mask(x)
        #seq_len = tf.shape(x)[1]

        #maxlen = tf.shape(x)[-1]
        maxlen = tf.shape(x)[1]
        positions = tf.range(start=0, limit=maxlen, delta=1)
        positions = self.pos_emb(positions)
        x = self.token_emb(x)
        x *= tf.math.sqrt(tf.cast(self.embed_dim, tf.float32))
        #x += self.pos_emb[:, :seq_len, :]

        x += positions

        #print(self.token_emb.mask)

        return x
    
    def compute_mask(self, *args, **kwargs):
        # return the padding mask from the inputs
        return self.token_emb.compute_mask(*args, **kwargs)

    def get_config(self):
        config = super().get_config()
        config.update({
            'token_emb': self.token_emb,
            'pos_emb': self.pos_emb,
            'vocab_size': self.vocab_size,
            'maxlen': self.maxlen,
            'embed_dim': self.embed_dim
        })
        return config
    
    #def compute_mask(self, inputs, mask=None):
        #if mask is None:
        #    print("NO MASK EMBEDDING")
        #    return None
    #    return tf.not_equal(inputs, 0)


class TransformerBlock(layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1, activation="gelu", **kwargs):
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

        #if mask == True:
        #input_shape = tf.shape(inputs)
        #batch_size = input_shape[0]
        #seq_len = input_shape[1]
        #padding_mask = create_padding_mask_2(inputs,batch_size, seq_len, seq_len, tf.bool)
        #padding_mask = causal_attention_mask(batch_size, seq_len, seq_len, tf.bool)
        #else:
            #padding_mask = None 
        
        #mask = padding_attention_mask_2(inputs)

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

def new_gelu(x):
    """
    Implementation of the GELU activation function currently in Google BERT repo (identical to OpenAI GPT).
    Reference: Gaussian Error Linear Units (GELU) paper: https://arxiv.org/abs/1606.08415
    """
    return 0.5 * x * (1.0 + tf.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * tf.pow(x, 3.0))))

class GPT_Block(layers.Layer):
    def __init__(self, embed_dim, num_heads, rate=0.1, use_causal_mask = False,  **kwargs):
        super(GPT_Block, self).__init__()

        self.mask=None
        self.use_causal_mask=use_causal_mask

        self.att = layers.MultiHeadAttention(num_heads, embed_dim)
        
        self.mlp = keras.Sequential(
            [layers.Dense(4*embed_dim, activation='gelu'), layers.Dense(embed_dim),layers.Dropout(rate)]
        )
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-5)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-5)


    #def create_padding_mask(self, seq):
    #    seq = tf.cast(tf.math.equal(seq, 0), tf.float32)
        # Add extra dimensions to add the padding
        # to the attention logits.
    #    return tf.expand_dims(tf.expand_dims(seq, axis=1), axis=1)

    def call(self, inputs, mask = None):
        #input_shape = tf.shape(inputs)
        #batch_size = input_shape[0]
        #seq_len = input_shape[1]
        
        #causal_mask = causal_attention_mask(batch_size, seq_len, seq_len, tf.bool)


        lnout1 = self.layernorm1(inputs)
        #attention_output = self.att(lnout1, lnout1, attention_mask=causal_mask)
        if self.use_causal_mask == False:
            #print("PADDING")
            #print(inputs.shape)
            #self.mask=padding_attention_mask_2(inputs)
            #self.mask = padding_attention_mask(inputs)
            #self.mask = self.create_padding_mask(inputs)
            #self.mask = padding_attention_mask_3(inputs)
            self.mask = mask
            #print("SHAPE")
            #print(self.mask.shape)

        attention_output = self.att(lnout1, lnout1, use_causal_mask=self.use_causal_mask, attention_mask=self.mask)

        lnout2 = self.layernorm2(inputs + attention_output)
        mlp_output = self.mlp(lnout2)
        x = lnout2 + mlp_output

        return x
    
    #def compute_mask(self, inputs, mask=None):
        #if mask is None:
        #    print("NO MASK")
        #    return None
    #    return tf.not_equal(inputs, 0)
    
    def get_config(self):
        config = super().get_config()
        config.update({
            'att': self.att,
            'mlp': self.mlp,
            'layernorm1': self.layernorm1,
            'layernorm2': self.layernorm2
        })
        return config