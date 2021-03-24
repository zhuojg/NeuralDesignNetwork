import tensorflow as tf
from tensorflow import keras
from models.graph import GraphTripleConvStack
from models.layers import build_mlp
import tensorflow_probability as tfp


class NDNRelation(keras.Model):
    def __init__(self, category_list, relation_list):
        super(NDNRelation, self).__init__()
        # the dimension of g_c and g_p are not same as supplementary
        # but i think this is right
        self.g_c = GraphTripleConvStack([(64, 512, 128), (128, 512, 128), (128, 512, 128), (128, 128, 32)]) 
        self.g_p = GraphTripleConvStack([(64 + 32, 512, 128), (128, 512, 128), (128, 512, 128), (128, 128, 128)])
        self.h_pred = build_mlp(
            dim_list=[
                128, 
                512, 
                len(relation_list)
            ],
            activation='leaky_relu',
            batch_norm='batch'
        )

    def call(self, obj_vecs, pred_gt_vecs, s_idx, o_idx, pred_vecs=None, training=True):
        """[summary]

        Args:
            objs ([type]): [description]
            triples_gt_dict ([type]): [description]

        Returns:
            
        """
        if training:
            assert pred_vecs is not None
        else:
            pred_vecs = pred_gt_vecs

        edges = tf.stack([s_idx, o_idx], axis=1)

        if training:
            # in paper
            # author did not describe how to get graph embedding with GCN
            # according to paper, the dimension of z is 32
            obj_vecs_with_gt, pred_vecs_with_gt = self.g_c(obj_vecs, pred_gt_vecs, edges, training=training)
            # (O, 32), (T, 32)
        else:
            # if not training
            # z should be sampled from N(0, 1)
            normal_0_1 = tfp.distributions.Normal(loc=0., scale=1.)
            obj_vecs_with_gt = normal_0_1.sample(sample_shape=(obj_vecs.shape[0], 32))
            pred_vecs_with_gt = normal_0_1.sample(sample_shape=(pred_vecs.shape[0], 32))

        # concat 
        new_obj_vecs = tf.concat([obj_vecs, obj_vecs_with_gt], axis=-1) # (O, 160)
        new_pred_vecs = tf.concat([pred_vecs, pred_vecs_with_gt], axis=-1) # (T, 160)

        _, new_pred_vecs = self.g_p(new_obj_vecs, new_pred_vecs, edges, training=training) 
        # (T, 128)

        # predicted class of every edge
        pred_cls = self.h_pred(new_pred_vecs, training=training) # (T, len(realtion_list) )
        pred_cls = tf.keras.layers.Softmax()(pred_cls)

        new_p = tf.math.argmax(pred_cls, axis=-1)

        result = {}
        result['new_p'] = new_p
        result['pred_cls'] = pred_cls
        result['obj_vecs_with_gt'] = obj_vecs_with_gt
        result['pred_vecs_with_gt'] = pred_vecs_with_gt

        return result
