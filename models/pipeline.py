from uuid import NAMESPACE_X500
from models.refinement import NDNRefinement
import tensorflow as tf
from tensorflow import keras

import tensorflow_probability as tfp

from models.relation import NDNRelation
from models.generation import NDNGeneration
from models.refinement import NDNRefinement

import os
import json
import math
import random
from PIL import Image, ImageDraw


class NeuralDesignNetwork:
    def __init__(self, category_list, pos_relation_list, size_relation_list, config, save=False, training=True):
        super(NeuralDesignNetwork, self).__init__()

        print(config)

        self.save = save
        self.training = training
        self.config = config
        self.category_list = category_list
        self.pos_relation_list = pos_relation_list
        self.size_relation_list = size_relation_list

        # construct vocab
        self.vocab = {
            'object_name_to_idx': {},
            'pos_pred_name_to_idx': {},
            'size_pred_name_to_idx': {}
        }

        self.vocab['object_name_to_idx']['__image__'] = 0
        self.vocab['pos_pred_name_to_idx']['__in_image__'] = 0
        self.vocab['size_pred_name_to_idx']['__in_image__'] = 0

        for idx, item in enumerate(category_list):
            self.vocab['object_name_to_idx'][item] = idx + 1
        
        for idx, item in enumerate(pos_relation_list):
            self.vocab['pos_pred_name_to_idx'][item] = idx + 1
        
        for idx, item in enumerate(size_relation_list):
            self.vocab['size_pred_name_to_idx'][item] = idx + 1

        # build GCN as described in supplementary material
        self.pos_relation = NDNRelation(category_list=self.vocab['object_name_to_idx'].keys(), relation_list=self.vocab['pos_pred_name_to_idx'].keys())
        self.size_relation = NDNRelation(category_list=self.vocab['object_name_to_idx'].keys(), relation_list=self.vocab['size_pred_name_to_idx'])

        self.generation = NDNGeneration()
        self.refinement = NDNRefinement()

        self.obj_embedding = keras.layers.Embedding(input_dim=len(self.vocab['object_name_to_idx']), output_dim=64)
        self.pos_pred_embedding = keras.layers.Embedding(input_dim=len(self.vocab['pos_pred_name_to_idx']), output_dim=64)
        self.size_pred_embedding = keras.layers.Embedding(input_dim=len(self.vocab['size_pred_name_to_idx']), output_dim=64)
        # define optimizer
        self.relation_optimizer = keras.optimizers.Adam(learning_rate=config['learning_rate'], beta_1=config['beta_1'], beta_2=config['beta_2'])
        self.generation_optimizer = keras.optimizers.Adam(learning_rate=config['learning_rate'], beta_1=config['beta_1'], beta_2=config['beta_2'])
        self.refinement_optimizer = keras.optimizers.Adam(learning_rate=config['learning_rate'], beta_1=config['beta_1'], beta_2=config['beta_2'])

        # define ckpt manger
        self.ckpt = tf.train.Checkpoint(
            obj_embedding=self.obj_embedding,
            pos_pred_embedding=self.pos_pred_embedding,
            size_pred_embedding=self.size_pred_embedding,
            pos_relation=self.pos_relation,
            size_relation=self.size_relation,
            generation=self.generation,
            refinement=self.refinement,
            relation_optimizer=self.relation_optimizer,
            generation_optimizer=self.generation_optimizer,
            refinement_optimizer=self.refinement_optimizer
        )

        # define training parameters
        self.iter_cnt = 0

    
    # define loss
    def relation_loss(self, pred_cls_gt, pred_cls_predicted, z):
        cls_loss = keras.losses.CategoricalCrossentropy()(pred_cls_gt, pred_cls_predicted)
        
        # TODO:
        # this part is a little confusing
        # according to paper, we need to calculate KL divergence of z and N(0, 1)
        # where `z` is calculated from g_p(G)
        # so we cannot get distribution of z

        # here i try to sample some data from N(0, 1)
        # then calculate discrete KL divergence
        # i'm not sure if this is correct
        normal_0_1 = tfp.distributions.Normal(loc=0., scale=1.)
        sample_from_normal_0_1 = normal_0_1.sample(sample_shape=(z.shape))
        KL_loss = keras.losses.KLDivergence()(sample_from_normal_0_1 ,z)

        return self.config['lambda_cls'] * cls_loss + self.config['lambda_kl_2'] * KL_loss

    def generation_loss(self, bb_gt, bb_predicted, mu, var, mu_prior, var_prior):
        # reconstruction loss is L1 loss
        recon_loss = tf.keras.losses.MeanAbsoluteError()(bb_gt, bb_predicted)

        # KL loss can be calculated according to mu and var
        sigma = tf.math.exp(0.5 * var)
        sigma_prior = tf.math.exp(0.5 * var_prior)
        KL_loss = tf.math.log(sigma_prior / sigma + 1e-8) + (tf.math.pow(sigma, 2) + tf.math.pow(mu - mu_prior, 2)) / (2 * tf.math.pow(sigma_prior, 2)) - 1./2

        return self.config['lambda_recon'] * recon_loss + self.config['lambda_kl_1'] * KL_loss

    @staticmethod
    def split_graph(objs, triples):
        # split triples, s, p and o all have size (T, 1)
        s, p, o = tf.split(triples, num_or_size_splits=3, axis=1)
        # squeeze, so the result size is (T,)
        s, p, o = [tf.squeeze(x, axis=1) for x in [s, p ,o]]

        return s, p, o


    def fetch_one_data(self, dataset):
        """[summary]

        Args:
            dataset ([type]): [description]
        
        Returns:
            objs (): 
            pos_triples ():
            size_triples ():
        """
        batch_json = dataset.take(1).as_numpy_iterator()

        for item in batch_json:
            batch_data = item

        '''
        combine the objects in one batch into a big graph
        '''
        obj_offset = 0
        all_obj = []
        all_boxes = []

        layout_height = 64.
        layout_width = 64.

        # triple: [s, p, o]
        # s: index in all_obj
        # p: index of relationship
        # o: index in all_obj
        all_pos_triples = []
        all_size_triples = []

        for item in batch_data:
            layout = json.load(open(item))
            cur_obj = []
            cur_boxes = []
            for category in layout.keys():
                for obj in layout[category]:
                    all_obj.append(self.vocab['object_name_to_idx'][category])
                    cur_obj.append(all_obj[-1])

                    x0, y0, x1, y1 = obj
                    x0 /= layout_width
                    y0 /= layout_height
                    x1 /= layout_width
                    y1 /= layout_height
                    all_boxes.append(tf.convert_to_tensor([x0, y0, x1, y1], dtype=tf.float32))
                    cur_boxes.append(all_boxes[-1])
            
            # at the end of one layout add __image__ item
            all_obj.append(self.vocab['object_name_to_idx']['__image__'])
            cur_obj.append(all_obj[-1])
            all_boxes.append(tf.convert_to_tensor([0, 0, 1, 1], dtype=tf.float32))
            cur_boxes.append(all_boxes[-1])

            # compute centers of layout in current layout
            obj_centers = []
            for box in cur_boxes:
                x0, y0, x1, y1 = box
                obj_centers.append([(x0 + x1) / 2, (y0 + y1) / 2])

            # calculate triples
            whole_image_idx = self.vocab['object_name_to_idx']['__image__']
            for obj_index, obj in enumerate(cur_obj):
                if obj == whole_image_idx:
                    continue

                # create a complete graph
                other_obj = [obj_idx for obj_idx, obj in enumerate(cur_obj) if (
                    obj_idx != obj_index and obj != whole_image_idx)]
                
                if len(other_obj) == 0:
                    continue

                for other in other_obj:
                    s = obj_index
                    o = other

                    sx0, sy0, sx1, sy1 = cur_boxes[s]
                    ox0, oy0, ox1, oy1 = cur_boxes[o]

                    d0 = obj_centers[s][0] - obj_centers[o][0]
                    d1 = obj_centers[s][1] - obj_centers[o][1]
                    theta = math.atan2(d1, d0)

                    # calculate position relationship
                    # now we have 6 kinds of position relationship
                    if sx0 < ox0 and sx1 > ox1 and sy0 < oy0 and sy1 > oy1:
                        p = 'surrounding'
                    elif sx0 > ox0 and sx1 < ox1 and sy0 > oy0 and sy1 < oy1:
                        p = 'inside'
                    elif theta >= 3 * math.pi / 4 or theta <= -3 * math.pi / 4:
                        p = 'left of'
                    elif -3 * math.pi / 4 <= theta < -math.pi / 4:
                        p = 'above'
                    elif -math.pi / 4 <= theta < math.pi / 4:
                        p = 'right of'
                    elif math.pi / 4 <= theta < 3 * math.pi / 4:
                        p = 'below'
                    p = self.vocab['pos_pred_name_to_idx'][p]

                    all_pos_triples.append([s + obj_offset, p, o + obj_offset])

                    # calculate size relationship
                    # now we have 3 kinds of size relationship
                    sw, sh = sx1 - sx0, sy1 - sy0
                    ow, oh = ox1 - ox0, oy1 - oy0
                    if sw > ow and sh > oh:
                        p = 'bigger'
                    elif sw < ow and sh < oh:
                        p = 'smaller'
                    elif sw * sh > ow * oh:
                        p = 'bigger'
                    elif sw * sh < ow * oh:
                        p = 'smaller'
                    else:
                        p = 'same'
                    p = self.vocab['size_pred_name_to_idx'][p]
                    all_size_triples.append([s + obj_offset, p, o + obj_offset])

            # add __in_image__ triples
            O = len(cur_obj)
            pos_in_image = self.vocab['pos_pred_name_to_idx']['__in_image__']
            size_in_image = self.vocab['size_pred_name_to_idx']['__in_image__']
            for i in range(O - 1):
                all_pos_triples.append([i + obj_offset, pos_in_image, O - 1 + obj_offset])
                all_size_triples.append([i + obj_offset, size_in_image, O - 1 + obj_offset])

            obj_offset += len(cur_obj)
        
        all_obj = tf.convert_to_tensor(all_obj)
        all_boxes = tf.convert_to_tensor(all_boxes)
        all_pos_triples = tf.convert_to_tensor(all_pos_triples)
        all_size_triples = tf.convert_to_tensor(all_size_triples)
            
        return all_obj, all_boxes, all_pos_triples, all_size_triples


    def run(self, config):
        train_dataset = tf.data.Dataset.list_files(os.path.join(config['data_dir'], '*.json'))
        train_dataset = train_dataset.repeat().shuffle(buffer_size=100).batch(batch_size=config['batch_size'])
        test_dataset = tf.data.Dataset.list_files(os.path.join(config['test_data_dir'], '*.json'))
        test_dataset = test_dataset.repeat().shuffle(buffer_size=100).batch(batch_size=config['batch_size'])
        sample_dataset = tf.data.Dataset.list_files(os.path.join(config['test_data_dir'], '*.json'))
        sample_dataset = test_dataset.repeat().shuffle(buffer_size=100).batch(batch_size=config['batch_size'])
        
        if self.save:
            ckpt_manager = tf.train.CheckpointManager(
                self.ckpt,
                config['checkpoint_dir'],
                max_to_keep=config['checkpoint_max_to_keep']
            )

            # init tensorboard writer
            train_log_dir = os.path.join(config['log_dir'], 'train')
            test_log_dir = os.path.join(config['log_dir'], 'test')
            train_summary_writer = tf.summary.create_file_writer(train_log_dir)
            test_summary_writer = tf.summary.create_file_writer(test_log_dir)

        # define metrics
        pos_relation_acc = keras.metrics.CategoricalAccuracy()
        size_relation_acc = keras.metrics.CategoricalAccuracy()
        recon_loss = keras.metrics.MeanAbsoluteError()

        # start training
        while self.iter_cnt < config['max_iteration_number']:
            objs, boxes, pos_triples_gt, size_triples_gt = self.fetch_one_data(dataset=train_dataset)

            result = self.run_step(objs, boxes, pos_triples_gt, size_triples_gt, training=True)

            pos_loss = result['pos_loss']
            size_loss = result['size_loss']
            gen_loss = result['gen_loss']
            pred_boxes = result['pred_boxes']
            gt_pos_cls = result['gt_pos_cls']
            pred_pos_cls = result['pred_pos_cls']
            gt_size_cls = result['gt_size_cls']
            pred_size_cls = result['pred_size_cls']

            pos_relation_acc.update_state(gt_pos_cls, pred_pos_cls)
            size_relation_acc.update_state(gt_size_cls, pred_size_cls)
            recon_loss.update_state(boxes, pred_boxes)

            print('Step: %d. Pos Loss: %f. Size Loss: %f. Position Classification Acc: %f. Size Classification Acc: %f' 
            % 
            (self.iter_cnt, pos_loss.numpy(), size_loss.numpy(), pos_relation_acc.result().numpy(), size_relation_acc.result()))

            print('Step: %d. Gen Loss: %f. Recon Loss: %f.'
            %
            (self.iter_cnt, gen_loss.numpy(), recon_loss.result().numpy()))

            if self.save:
                with train_summary_writer.as_default():
                    tf.summary.scalar('pos_loss', pos_loss, step=self.iter_cnt)
                    tf.summary.scalar('size_loss', size_loss, step=self.iter_cnt)
                    tf.summary.scalar('pos_relation_acc', pos_relation_acc.result(), step=self.iter_cnt)
                    tf.summary.scalar('size_relation_acc', size_relation_acc.result(), step=self.iter_cnt)

                    tf.summary.scalar('gen_loss', gen_loss, step=self.iter_cnt)
                    tf.summary.scalar('recon_loss', recon_loss.result(), step=self.iter_cnt)
            
            pos_relation_acc.reset_states()
            size_relation_acc.reset_states()
            recon_loss.reset_states()

            # train refinement
            # TODO

            if self.save and (self.iter_cnt + 1) % config['checkpoint_every'] == 0:
                ckpt_manager.save()
                print('Checkpoint saved.')
            
            """
            start testing
            """
            objs, boxes, pos_triples_gt, size_triples_gt = self.fetch_one_data(dataset=test_dataset)

            result = self.run_step(objs, boxes, pos_triples_gt, size_triples_gt, training=False)
            
            pred_boxes = result['pred_boxes']
            gt_pos_cls = result['gt_pos_cls']
            pred_pos_cls = result['pred_pos_cls']
            gt_size_cls = result['gt_size_cls']
            pred_size_cls = result['pred_size_cls']

            pos_relation_acc.update_state(gt_pos_cls, pred_pos_cls)
            size_relation_acc.update_state(gt_size_cls, pred_size_cls)
            recon_loss.update_state(boxes, pred_boxes)

            if self.save:
                with test_summary_writer.as_default():
                    tf.summary.scalar('pos_loss', pos_loss, step=self.iter_cnt)
                    tf.summary.scalar('size_loss', size_loss, step=self.iter_cnt)
                    tf.summary.scalar('pos_relation_acc', pos_relation_acc.result(), step=self.iter_cnt)
                    tf.summary.scalar('size_relation_acc', size_relation_acc.result(), step=self.iter_cnt)

                    tf.summary.scalar('gen_loss', gen_loss, step=self.iter_cnt)
                    tf.summary.scalar('recon_loss', recon_loss.result(), step=self.iter_cnt)
            
            pos_relation_acc.reset_states()
            size_relation_acc.reset_states()
            recon_loss.reset_states()

            """
            start sampling
            """
            for idx in range(10):
                objs, boxes, pos_triples_gt, size_triples_gt = self.fetch_one_data(dataset=sample_dataset)

                result = self.run_step(objs, boxes, pos_triples_gt, size_triples_gt, training=False)
                
                pred_boxes = result['pred_boxes']
                gt_pos_cls = result['gt_pos_cls']
                pred_pos_cls = result['pred_pos_cls']
                gt_size_cls = result['gt_size_cls']
                pred_size_cls = result['pred_size_cls']

                self.draw_boxes(
                    objs, 
                    pred_boxes, 
                    os.path.join(
                        self.config['train_sample_dir'], 'train_%d_%d.png' 
                        % (self.iter_cnt, idx)
                    )
                )


            self.iter_cnt += 1

    def run_step(self, config, objs, boxes, pos_triples_gt, size_triples_gt, training=True):
        step_result = {}

        s, pos_pred_gt, o = self.split_graph(objs, pos_triples_gt)
        _, size_pred_gt, o = self.split_graph(objs, size_triples_gt)

        # randomly mask to generate training data
        pos_pred = pos_pred_gt.numpy()
        for item, idx in enumerate(pos_pred):
            if random.random() <= config['mask_rate']:
                pos_pred[idx] = len(self.vocab['pos_pred_name_to_idx']) - 1

        size_pred = size_pred_gt.numpy()
        for item, idx in enumerate(size_pred):
            if random.random() <= config['mask_rate']:
                size_pred[idx] = len(self.vocab['size_pred_name_to_idx']) - 1

        pos_pred = tf.convert_to_tensor(pos_pred)
        size_pred = tf.convert_to_tensor(size_pred)

        if training:
            # train pos relation
            with tf.GradientTape(persistent=True) as tape:
                # get embedding of obj and pred
                obj_vecs = self.obj_embedding(objs)
                pred_vecs = self.pos_pred_embedding(pos_pred)
                pred_gt_vecs = self.pos_pred_embedding(pos_pred_gt)

                result = self.pos_relation(obj_vecs, pred_gt_vecs, s, o, pred_vecs=pred_vecs, training=True)

                # embedding pred with one_hot, to calculate cross entropy loss
                pred_gt_one_hot = tf.one_hot(pos_pred_gt, depth=len(self.vocab['pos_pred_name_to_idx']))
                step_result['gt_pos_cls'] = pred_gt_one_hot
                # get latent variable of G_gt
                z = tf.concat([result['obj_vecs_with_gt'], result['pred_vecs_with_gt']], axis=0)
                pos_loss = self.relation_loss(pred_gt_one_hot, result['pred_cls'], z)
                step_result['pos_loss'] = pos_loss
                step_result['pred_pos_cls'] = result['pred_cls']

            train_var = self.pos_relation.trainable_variables \
                        + self.obj_embedding.trainable_variables + self.pos_pred_embedding.trainable_variables
            gradients = tape.gradient(pos_loss, train_var)

            self.relation_optimizer.apply_gradients(
                zip(gradients, train_var)
            )

            # train size relation
            with tf.GradientTape(persistent=True) as tape:
                # TODO:
                # use the same obj embedding here
                # but when to update the weight of obj embedding is not sure
                obj_vecs = self.obj_embedding(objs)
                pred_vecs = self.size_pred_embedding(size_pred)
                pred_gt_vecs = self.size_pred_embedding(size_pred_gt)

                result = self.size_relation(obj_vecs, pred_gt_vecs, s, o, pred_vecs=pred_vecs, training=True)

                pred_gt_one_hot = tf.one_hot(size_pred_gt, depth=len(self.vocab['size_pred_name_to_idx']))
                step_result['gt_size_cls'] = pred_gt_one_hot
                z = tf.concat([result['obj_vecs_with_gt'], result['pred_vecs_with_gt']], axis=0)
                size_loss = self.relation_loss(pred_gt_one_hot, result['pred_cls'], z)
                step_result['size_loss'] = size_loss
                step_result['pred_size_cls'] = result['pred_cls']

            train_var = self.size_relation.trainable_variables \
                        + self.obj_embedding.trainable_variables + self.size_pred_embedding.trainable_variables
            gradients = tape.gradient(size_loss, train_var)

            self.relation_optimizer.apply_gradients(
                zip(gradients, train_var)
            )
        
        else:
            pos_pred = tf.ones_like(pos_pred_gt) * (len(self.vocab['pos_pred_name_to_idx']) - 1)
            size_pred = tf.ones_like(size_pred_gt) * (len(self.vocab['size_pred_name_to_idx']) - 1)

            obj_vecs = self.obj_embedding(objs)

            pred_vecs = self.pos_pred_embedding(pos_pred)
            result = self.pos_relation(obj_vecs, pred_vecs, s, o, training=False)
            pred_gt_one_hot = tf.one_hot(pos_pred_gt, depth=len(self.vocab['pos_pred_name_to_idx']))
            step_result['gt_pos_cls'] = pred_gt_one_hot
            step_result['pred_pos_cls'] = result['pred_cls']

            pred_vecs = self.size_pred_embedding(size_pred)
            result = self.pos_relation(obj_vecs, pred_vecs, s, o, training=False)
            pred_gt_one_hot = tf.one_hot(size_pred_gt, depth=len(self.vocab['size_pred_name_to_idx']))
            step_result['gt_size_cls'] = pred_gt_one_hot
            step_result['pred_size_cls'] = result['pred_cls']



        # train generation
        s, pos_pred, o = self.split_graph(objs, pos_triples_gt)
        _, size_pred, _ = self.split_graph(objs, size_triples_gt)
        
        if training:
            with tf.GradientTape(persistent=True) as tape:
                obj_vecs = self.obj_embedding(objs)
                pos_pred_vecs = self.pos_pred_embedding(pos_pred)
                size_pred_vecs = self.size_pred_embedding(size_pred)

                # concat pos_pred and size_pred
                pred_vecs = tf.concat([pos_pred_vecs, size_pred_vecs], axis=0)
                s_idx = tf.concat([s, s], axis=0)
                o_idx = tf.concat([o, o], axis=0)

                result = self.generation(obj_vecs, pred_vecs, boxes, s_idx, o_idx, training=True)
                step_result['pred_boxes'] = result['pred_boxes']
                # `result` contains output of k iterations
                gen_loss = self.generation_loss(
                    bb_gt=boxes, 
                    bb_predicted=result['pred_boxes'],
                    mu=result['mu'],
                    var=result['var'],
                    mu_prior=result['mu_prior'],
                    var_prior=result['var_prior']
                )
                step_result['gen_loss'] = gen_loss
            
            train_var = self.generation.trainable_variables
            gradients = tape.gradient(gen_loss, train_var)
            
            self.generation_optimizer.apply_gradients(
                zip(gradients, train_var)
            )
        
        else:
            # when not traing
            # use the relationship predicted by previous model
            pos_pred = tf.math.argmax(result['pred_pos_cls'], axis=1)
            size_pred = tf.math.argmax(result['pred_size_cls'], axis=1)

            obj_vecs = self.obj_embedding(objs)
            pos_pred_vecs = self.pos_pred_embedding(pos_pred)
            size_pred_vecs = self.size_pred_embedding(size_pred)

            # concat pos_pred and size_pred
            pred_vecs = tf.concat([pos_pred_vecs, size_pred_vecs], axis=0)
            s_idx = tf.concat([s, s], axis=0)
            o_idx = tf.concat([o, o], axis=0)

            result = self.generation(obj_vecs, pred_vecs, boxes, s_idx, o_idx, training=False)
            step_result['pred_boxes'] = result['pred_boxes']


        return step_result
    
    def draw_boxes(self, obj_cls, boxes, output_path):
        assert len(obj_cls) == len(boxes)

        colormap = ['#aaaaaa','#0000ff', '#00ff00', '#00ffff', '#ff0000', '#ff00ff', '#ffff00', '#ffffff']
        
        CANVA_SIZE = 640
        canva = Image.new('RGB', (CANVA_SIZE, CANVA_SIZE), (0, 0, 0))
        draw = ImageDraw.Draw(canva)

        for idx in range(len(obj_cls)):
            temp_cls = obj_cls[idx]
            x, y, w, h = boxes[idx]

            x0 = x
            y0 = y
            x1 = x + w
            y1 = y + h
            
            draw.rectangle([x0, y0, x1, y1], outline=colormap[temp_cls])

        canva.save(output_path)
