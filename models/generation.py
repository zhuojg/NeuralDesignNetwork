import tensorflow as tf
from tensorflow import keras
from models.graph import GraphTripleConvStack
from models.layers import build_mlp
import tensorflow_probability as tfp


class NDNGeneration(keras.Model):
    def __init__(self):
        super(NDNGeneration, self).__init__()
        self.g_enc = GraphTripleConvStack([(64, 512, 128), (128, 512, 128), (128, 512, 128)])
        # self.g_update = GraphTripleConvStack([(128, 512, 128)])
        self.h_bb_dec = build_mlp(dim_list=[32 + 128, 128, 64, 4])

        # TODO:
        # in paper, g_update is a GCN with node feature and bbox as input
        # howevert, I think it's not possible to handle this input by GCN described in paper
        # so I rewrite g_update as a MLP
        g_update_f_input = keras.layers.Input(shape=(128))
        g_update_b_input = keras.layers.Input(shape=(4))
        g_update_input = tf.concat([g_update_f_input, g_update_b_input], axis=-1)
        g_update_hidden = keras.layers.Dense(128, activation=tf.nn.leaky_relu)(g_update_input)
        self.g_update_embedding = keras.Model(inputs=[g_update_f_input, g_update_b_input], outputs=[g_update_hidden])
        self.g_update = GraphTripleConvStack([(128, 512, 128)])

        # build h_bb_encoder
        # h_bb_encoder take condition and bb_gt as input
        h_bb_enc_inputs_bb = keras.layers.Input(shape=(4))
        h_bb_enc_inputs_c = keras.layers.Input(shape=(128))
        h_bb_enc_hidden_bb = keras.layers.Dense(128, activation=tf.nn.leaky_relu)(h_bb_enc_inputs_bb)
        h_bb_enc_hidden_c = keras.layers.Dense(128, activation=tf.nn.leaky_relu)(h_bb_enc_inputs_c)
        h_bb_enc_hidden = tf.concat([h_bb_enc_hidden_bb, h_bb_enc_hidden_c], axis=-1)
        h_bb_enc_result = keras.layers.Dense(32, activation=tf.nn.leaky_relu)(h_bb_enc_hidden)
        h_bb_enc_mu = keras.layers.Dense(32, activation=tf.nn.leaky_relu)(h_bb_enc_result)
        h_bb_enc_var = keras.layers.Dense(32, activation=tf.nn.leaky_relu)(h_bb_enc_result)
        self.h_bb_enc = keras.Model(inputs=[h_bb_enc_inputs_bb, h_bb_enc_inputs_c], outputs=[h_bb_enc_mu, h_bb_enc_var])

        # prior_encoder take condition as input
        # and we try to minimize the KL divergence between prior_encoder and h_bb_encoder
        # so during inference, we can generate bb with condition only
        prior_input = keras.layers.Input(shape=(128))
        prior_x = keras.layers.Dense(128, activation=tf.nn.leaky_relu)(prior_input)
        prior_x = keras.layers.Dense(128, activation=tf.nn.leaky_relu)(prior_x)
        prior_x = keras.layers.Dense(32, activation=tf.nn.leaky_relu)(prior_x)
        prior_mu = keras.layers.Dense(32, activation=tf.nn.leaky_relu)(prior_x)
        prior_var = keras.layers.Dense(32, activation=tf.nn.leaky_relu)(prior_x)
        self.prior_encoder = keras.Model(inputs=[prior_input], outputs=[prior_mu, prior_var])
    
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

    def call(self, objs, obj_vecs, pred_vecs, boxes, s_idx, o_idx, training=True):
        O = obj_vecs.shape[0]

        result = {}
        result['pred_boxes'] = []
        result['mu'] = []
        result['var'] = []
        result['mu_prior'] = []
        result['var_prior'] = []

        new_obj_vecs, new_pred_vecs = self.g_enc(obj_vecs, pred_vecs, tf.stack([s_idx, o_idx], axis=1), training=training)
        # (O, 128)

        layout_size_list = []
        layout_pred_size_list = []
        temp_layout = []

        for item in objs:
            if item != 0:
                temp_layout.append(item)
            else:
                layout_size_list.append(len(temp_layout) + 1)
                temp_layout = []

                # with N elements in layout
                # we have N - 1 elements except __image__
                # these N - 1 elements are all connected to each other
                # so we have (N-1)(N-2) preds
                # for __image__ item, it is connected to all other N - 1 elements
                # so the final pred number is (N - 1)(N- 2) + (N - 1) = (N - 1)^2
                layout_pred_size_list.append((layout_size_list[-1] - 1) ** 2)
        
        # simulate k iteration
        # we have batch_size layout in objs
        # we need to calculate c_k for different layout
        # layout is marked by __image__ item, 
        # which is at the end of a list of elements
        obj_offset = 0
        pred_offset = 0
        for idx, layout_size in enumerate(layout_size_list):
            previous_bb = []
            temp_obj_vecs = new_obj_vecs[obj_offset : obj_offset + layout_size]
            temp_pred_vecs = new_pred_vecs[pred_offset : pred_offset + layout_pred_size_list[idx]]
            temp_s_idx = s_idx[pred_offset : pred_offset + layout_pred_size_list[idx]] - obj_offset
            temp_o_idx = o_idx[pred_offset : pred_offset + layout_pred_size_list[idx]] - obj_offset

            for k in range(layout_size):
                temp_bb = previous_bb.copy()
                while len(temp_bb) < layout_size:
                    temp_bb.append([0., 0., 0., 0.])
                
                temp_new_obj_vecs = self.g_update_embedding([temp_obj_vecs, tf.convert_to_tensor(temp_bb)], training=training)
                temp_new_obj_vecs, temp_new_pred_vecs = self.g_update(temp_new_obj_vecs, temp_pred_vecs, tf.stack([temp_s_idx, temp_o_idx], axis=1), training=training)

                c_k = tf.reduce_mean(tf.concat([temp_new_obj_vecs, temp_new_pred_vecs], axis=0), axis=0)

                if training:
                    temp_boxes = tf.expand_dims(boxes[k + obj_offset], axis=0)
                    c_k = tf.expand_dims(c_k, axis=0)
                    z_mu, z_var = self.h_bb_enc([temp_boxes, c_k], training=training)
                    result['mu'].append(z_mu)
                    result['var'].append(z_var)
                    z = self.reparameterize(z_mu, z_var)

                    z_c_k = tf.concat([z, c_k], axis=-1)
                    bb_k_predicted = self.h_bb_dec(z_c_k, training=True)
                    bb_k_predicted = tf.squeeze(bb_k_predicted, axis=0)
                    result['pred_boxes'].append(bb_k_predicted)

                    z_mu, z_var = self.prior_encoder(c_k)
                    result['mu_prior'].append(z_mu)
                    result['var_prior'].append(z_var)
                    previous_bb.append(boxes[k + obj_offset])

                else:
                    c_k = tf.expand_dims(c_k, axis=0)
                    z_mu, z_var = self.prior_encoder(c_k)
                    z = self.reparameterize(z_mu, z_var)
                    z_c_k = tf.concat([z, c_k], axis=-1)
                    bb_k_predicted = self.h_bb_dec(z_c_k, training=False)
                    bb_k_predicted = tf.squeeze(bb_k_predicted, axis=0)
                    result['pred_boxes'].append(bb_k_predicted)

                    # for __image__ item, use gt bbox
                    if objs[obj_offset + k] == 0:
                        previous_bb.append([0., 0., 1., 1.,])
                    else:
                        previous_bb.append(bb_k_predicted)
                
            # to maintain the size of predicted boxes
            # add bbox for __image__ item
            obj_offset += (layout_size)
            pred_offset += (layout_pred_size_list[idx])
        
        return result
