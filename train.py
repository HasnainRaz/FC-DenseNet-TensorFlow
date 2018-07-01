import tensorflow as tf
import utility
from model import DenseTiramisu
import os
import cv2
import numpy as np


class TrainEval(object):

    def __init__(self, image_paths, mask_paths, eval_image_paths, eval_mask_paths, model_save_dir, num_classes):
        self.train_image_paths = image_paths
        self.train_mask_paths = mask_paths
        self.eval_image_paths = eval_image_paths
        self.eval_mask_paths = eval_mask_paths
        self.num_classes = num_classes
        self.num_train_images = len(self.train_image_paths)
        self.num_eval_images = len(self.eval_image_paths)
        self.model_save_dir = model_save_dir

    def train_eval(self, batch_size, growth_k, layers_per_block, epochs, learning_rate=1e-3):
        """Trains the model on the dataset, and does periodic validations."""
        train_data, train_queue_init = utility.data_batch(
            self.train_image_paths, self.train_mask_paths, batch_size)
        train_image_tensor, train_mask_tensor = train_data

        eval_data, eval_queue_init = utility.data_batch(
            self.eval_image_paths, self.eval_mask_paths, batch_size)
        eval_image_tensor, eval_mask_tensor = eval_data

        image_ph = tf.placeholder(tf.float32, shape=[None, 256, 256, 3])
        mask_ph = tf.placeholder(tf.int32, shape=[None, 256, 256, 1])
        training = tf.placeholder(tf.bool, shape=[])

        tiramisu = DenseTiramisu(growth_k, layers_per_block, self.num_classes)

        logits = tiramisu.model(image_ph, training)

        loss = tf.reduce_mean(tiramisu.xentropy_loss(logits, mask_ph))

        with tf.variable_scope("mean_iou_train"):
            iou, iou_update = tiramisu.calculate_iou(mask_ph, logits)

        optimizer = tf.train.AdamOptimizer(learning_rate)
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            opt = optimizer.minimize(loss)

        running_vars = tf.get_collection(
            tf.GraphKeys.LOCAL_VARIABLES, scope="mean_iou_train")

        reset_iou = tf.variables_initializer(var_list=running_vars)

        saver = tf.train.Saver(max_to_keep=20)
        encoder_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='encoder')
        print('Encoder Variables: ', encoder_vars)
        with tf.Session() as sess:
            sess.run([tf.global_variables_initializer(),
                      tf.local_variables_initializer()])
            for epoch in range(epochs):
                writer = tf.summary.FileWriter(self.model_save_dir, sess.graph)
                sess.run([train_queue_init, eval_queue_init])
                total_train_cost, total_eval_cost = 0, 0
                total_train_iou, total_eval_iou = 0, 0
                for train_step in range(self.num_train_images // batch_size):
                    image_batch, mask_batch, _ = sess.run(
                        [train_image_tensor, train_mask_tensor, reset_iou])
                    # print("Mask batch shape:", mask_batch.shape)
                    feed_dict = {image_ph: image_batch,
                                 mask_ph: mask_batch,
                                 training: True}
                    cost, _, _ = sess.run(
                        [loss, opt, iou_update], feed_dict=feed_dict)
                    train_iou = sess.run(iou, feed_dict=feed_dict)
                    total_train_cost += cost
                    total_train_iou += train_iou
                    if train_step % 50 == 0:
                        print("Step: ", train_step, "Cost: ",
                              cost, "IoU:", train_iou)

                for eval_step in range(self.num_eval_images // batch_size):
                    image_batch, mask_batch, _ = sess.run(
                        [eval_image_tensor, eval_mask_tensor, reset_iou])
                    feed_dict = {image_ph: image_batch,
                                 mask_ph: mask_batch,
                                 training: True}
                    eval_cost, _ = sess.run(
                        [loss, iou_update], feed_dict=feed_dict)
                    eval_iou = sess.run(iou, feed_dict=feed_dict)
                    total_eval_cost += eval_cost
                    total_eval_iou += eval_iou

                print("Epoch: ", epoch, "train loss: ", total_train_cost / train_step, "eval loss: ",
                      total_eval_cost)
                print("Epoch: ", epoch, "train eval: ", total_train_iou / train_step, "eval iou: ",
                      total_eval_iou)

                print("Saving model...")
                saver.save(sess, self.model_save_dir, global_step=epoch)

    def infer(self, image_paths, batch_size):
        infer_data, infer_queue_init = utility.data_batch(
            image_paths, None, batch_size)
        image_ph = tf.placeholder(tf.float32, shape=[None, 256, 256, 3])
        training = tf.placeholder(tf.bool, shape=[])
        tiramisu = DenseTiramisu(16, [2, 3, 3], 2)
        logits = tiramisu.model(image_ph, training)
        mask = tf.squeeze(tf.argmax(logits, axis=3))

        saver = tf.train.Saver()
        with tf.Session() as sess:
            saver.restore(sess, 'trained_tiramisu/model.ckpt-18')
            sess.run(infer_queue_init)
            for i in range(len(image_paths) // batch_size):
                image = sess.run(infer_data)
                feed_dict = {
                    image_ph: image,
                    training: True
                }
                prediction = sess.run(mask, feed_dict)
                for j in range(prediction.shape[0]):
                    cv2.imwrite('predictions/{}.png'.format(j), 255 * prediction[j, :, :])
