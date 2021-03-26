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
        node_input = keras.layers.Input(shape=(64))
        bb_input = keras.layers.Input(shape=(4))
        node_bb_input = tf.concat([node_input, bb_input], axis=-1)
        node_bb_feature = keras.layers.Dense(64, activation=tf.nn.leaky_relu)(node_bb_input)
        self.node_bb_embedding = keras.Model(inputs=[node_input, bb_input], outputs=[node_bb_feature])

        self.g_ft = GraphTripleConvStack([(64, 512, 128), (128, 512, 128), (128,512, 128), (128, 128, 128)])
    
        node_feature = keras.layers.Input(shape=(128))
        predicted_bb = keras.layers.Dense(4, activation=tf.nn.leaky_relu)(node_feature)
        self.node_to_bb = keras.Model(inputs=node_feature, outputs=[predicted_bb])


    def call(self, obj_vecs, pred_vecs, pred_boxes, s_idx, o_idx, training=True):
        result = {}

        new_obj_vecs = self.node_bb_embedding([obj_vecs, pred_boxes], training=training)
        new_obj_vecs, _ = self.g_ft(new_obj_vecs, pred_vecs, tf.stack([s_idx, o_idx], axis=1), training=training)

        bb_predicted = self.node_to_bb(new_obj_vecs, training=training)
        result['bb_predicted'] = bb_predicted

        return result
