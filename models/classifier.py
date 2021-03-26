import tensorflow as tf
from tensorflow import keras
from tensorflow.python.ops.op_selector import graph_inputs
from models.graph import GraphTripleConvStack
from models.layers import build_mlp


class LayoutClassifier(keras.Model):
    """[summary]

    4 GCN layers
    3 fully connected layer

    """
    def __init__(self):
        super(LayoutClassifier, self).__init__()
        
        node_input = keras.layers.Input(shape=(64))
        bb_input = keras.layers.Input(shape=(4))
        node_bb_input = tf.concat([node_input, bb_input], axis=-1)
        node_bb_feature = keras.layers.Dense(64, activation='relu')(node_bb_input)
        self.node_bb_embedding = keras.Model(inputs=[node_input, bb_input], outputs=[node_bb_feature])

        self.g_encoder = GraphTripleConvStack([(64, 512, 128), (128, 512, 128), (128, 512, 128), (128, 128, 128)])

        scorer_input = keras.layers.Input(shape=(128))
        scorer_hidden = keras.layers.Dense(512, activation='relu')(scorer_input)
        scorer_feature = keras.layers.Dense(128, activation='relu')(scorer_hidden)
        scorer_result = keras.layers.Dense(2)(scorer_feature)
        scorer_result = keras.layers.Softmax()(scorer_result)
        self.scorer = keras.Model(inputs=[scorer_input], outputs=[scorer_result, scorer_feature])
    
    def call(self, inputs, training=True):
        result = {}

        obj_vecs = inputs['obj_vecs']
        pred_vecs = inputs['pred_vecs']
        s_idx = inputs['s_idx']
        o_idx = inputs['o_idx']
        boxes = inputs['boxes']

        edges = tf.stack([s_idx, o_idx], axis=1)

        new_obj_vecs = self.node_bb_embedding([obj_vecs, boxes])

        new_obj_vecs, new_pred_vecs = self.g_encoder(new_obj_vecs, pred_vecs, edges, training=training)

        # predicted_bb = self.node_to_bb(new_obj_vecs)
        
        graph_feature = tf.concat([new_obj_vecs, new_pred_vecs], axis=-1)
        graph_feature = tf.reduce_mean(graph_feature, axis=0)

        scorer_result, scorer_feature = self.scorer([graph_feature])

        result['score'] = scorer_result
        result['feature'] = scorer_feature

        return result
