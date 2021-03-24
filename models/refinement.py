import tensorflow as tf
from tensorflow import keras
from models.graph import GraphTripleConvStack
from models.layers import build_mlp
import tensorflow_probability as tfp


class NDNRefinement(keras.Model):
    def __init__(self):
        super(NDNRefinement, self).__init__()
        # TODO:
        # in paper, g_ft is a GCN
        # however, as described in generation.py
        # GCN cannot take graph and bbox as input (I donot know how)
        # so rewrite it as a MLP
        self.g_ft = GraphTripleConvStack([(64, 512, 128), (128, 512, 128), (128,512, 128), (128, 128, 128)])
        
        # use encoder to concat g_ft output and bbox
        encoder_input_b = keras.layers.Input(shape=(4))
        encoder_input_f = keras.layers.Input(shape=(128))
        encoder_input = tf.concat([encoder_input_b, encoder_input_f], axis=-1)
        encoder_hidden = keras.layers.Dense(128, activation=tf.nn.leaky_relu)(encoder_input)
        encoder_hidden = keras.layers.Dense(128, activation=tf.nn.leaky_relu)(encoder_hidden)
        encoder_result = keras.layers.Dense(4, activation=tf.nn.leaky_relu)(encoder_hidden)
        self.encoder = keras.Model(inputs=[encoder_input_b, encoder_input_f], outputs=[encoder_result])
    
    def call(self, obj_vecs, pred_vecs, pred_boxes, s_idx, o_idx, training=True):
        result = {}

        new_obj_vecs, _ = self.g_ft(obj_vecs, pred_vecs, tf.stack([s_idx, o_idx], axis=1), training=training)

        bb_predicted = self.encoder([pred_boxes, new_obj_vecs], training=training)

        result['bb_predicted'] = bb_predicted

        return result
