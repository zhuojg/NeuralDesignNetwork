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
        
        z_encoder_input = keras.layers.Input(shape=(32))
        z_mu = keras.layers.Dense(32)(z_encoder_input)
        z_var = keras.layers.Dense(32)(z_encoder_input)
        self.z_encoder = keras.Model(inputs=[z_encoder_input], outputs=[z_mu, z_var])

        self.g_p = GraphTripleConvStack([(64, 512, 128), (128, 512, 128), (128, 512, 128), (128, 128, 128)])
        self.h_pred = build_mlp(
            dim_list=[
                128, 
                512, 
                len(relation_list)
            ],
            activation='leaky_relu',
            batch_norm='batch'
        )

        node_input = keras.layers.Input(shape=(64))
        z_input = keras.layers.Input(shape=(32))
        node_embedding_input = tf.concat([node_input, z_input], axis=-1)
        node_embedding_output = keras.layers.Dense(64, activation=tf.nn.leaky_relu)(node_embedding_input)
        self.node_embedding = keras.Model(inputs=[node_input, z_input], outputs=[node_embedding_output])

    def reparameterize(self, mu, var):
        """reparameterize trick for VAE

        sampling operation creates a bottleneck 
        because backpropagation cannot flow through a random node

        Args:
            mu: mean of Normal distribution
            var: std of Normal distribution

        Returns:
            z: a sample result
        """
        eps = tf.random.normal(shape=mu.shape)
        z = eps * tf.exp(var * .5) + mu
        return z

    def call(self, obj_vecs, pred_gt_vecs, s_idx, o_idx, pred_vecs=None, training=True):
        """[summary]

        Args:
            objs ([type]): [description]
            triples_gt_dict ([type]): [description]

        Returns:
            
        """
        result = {}

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
            z_mu, z_var = self.z_encoder([obj_vecs_with_gt])
            z = self.reparameterize(z_mu, z_var)
            # (O, 32), (T, 32)

            result['z_mu'] = z_mu
            result['z_var'] = z_var
        else:
            # if not training
            # z should be sampled from N(0, 1)
            normal_0_1 = tfp.distributions.Normal(loc=0., scale=1.)
            # obj_vecs_with_gt = normal_0_1.sample(sample_shape=(obj_vecs.shape[0], 32))
            # pred_vecs_with_gt = normal_0_1.sample(sample_shape=(pred_vecs.shape[0], 32))
            z = normal_0_1.sample(sample_shape=(obj_vecs.shape[0], 32))

        # concat 

        new_obj_vecs = self.node_embedding([obj_vecs, z])

        _, new_pred_vecs = self.g_p(new_obj_vecs, pred_vecs, edges, training=training)

        # predicted class of every edge
        pred_cls = self.h_pred(new_pred_vecs, training=training) # (T, len(realtion_list) )
        pred_cls = tf.keras.layers.Softmax()(pred_cls)

        new_p = tf.math.argmax(pred_cls, axis=-1)

        result['new_p'] = new_p
        result['pred_cls'] = pred_cls

        return result
