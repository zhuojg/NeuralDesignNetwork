import tensorflow as tf
from tensorflow import keras
from models.graph import GraphTripleConvStack
from models.layers import build_mlp
import tensorflow_probability as tfp


class NDNGeneration(keras.Model):
    def __init__(self):
        super(NDNGeneration, self).__init__()
        self.g_enc = GraphTripleConvStack([(64, 512, 128), (128, 512, 128), (128, 512, 128)])
        self.g_update = GraphTripleConvStack([128, 512, 128])
        self.h_bb_dec = build_mlp(dim_list=[32 + 128, 128, 64, 4])

        # build h_bb_encoder
        h_bb_enc_inputs_bb = keras.layers.Input(shape=(4))
        h_bb_enc_inputs_c = keras.layers.Input(shape=(128))
        h_bb_enc_hidden_bb = keras.layers.Dense(128)(h_bb_enc_inputs_bb)
        h_bb_enc_hidden_c = keras.layers.Dense(128)(h_bb_enc_inputs_c)
        h_bb_enc_hidden = tf.concat([h_bb_enc_hidden_bb, h_bb_enc_hidden_c], axis=-1)
        h_bb_enc_result = keras.layers.Dense(32)(h_bb_enc_hidden)
        h_bb_enc_mu = keras.layers.Dense(32)(h_bb_enc_result)
        h_bb_enc_var = keras.layers.Dense(32)(h_bb_enc_result)
        self.h_bb_enc = keras.Model(inputs=[h_bb_enc_inputs_bb, h_bb_enc_inputs_c], outputs=[h_bb_enc_mu, h_bb_enc_var])

    
    def call(self, obj_vecs, pred_vecs):
        pass
