import os
import cv2
import utility
import tensorflow as tf
from helpers import get_data_paths_list


class DenseTiramisu(object):
    """
    This class forms the Tiramisu model for segmentation of input images.
    """
    def __init__(self, growth_k, layers_per_block, num_classes):
        """
        Initializes the Tiramisu based on the specified parameters.
        Args:
            growth_k: Integer, growth rate of the Tiramisu.
            layers_per_block: List of integers, the number of layers in each dense block.
            num_classes: Integer: Number of classes to segment.
        """
        self.growth_k = growth_k
        self.layers_per_block = layers_per_block
        self.nb_blocks = len(layers_per_block)
        self.num_classes = num_classes
        self.logits = None

    def xentropy_loss(self, logits, labels):
        """
        Calculates the cross-entropy loss over each pixel in the ground truth
        and the prediction.
        Args:
            logits: Tensor, raw unscaled predictions from the network.
            labels: Tensor, the ground truth segmentation mask.

        Returns:
            loss: The cross entropy loss over each image in the batch.
        """
        labels = tf.cast(labels, tf.int32)
        logits = tf.reshape(logits, [tf.shape(logits)[0], -1, self.num_classes])
        labels = tf.reshape(labels, [tf.shape(labels)[0], -1])
        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
                logits=logits, labels=labels, name="loss")

        return loss

    def calculate_iou(self, mask, prediction):
        """
        Calculates the mean intersection over union (mean pixel accuracy)
        Args:
            mask: Tensor, The ground truth input segmentation mask.
            prediction: Tensor, the raw unscaled prediction from the network.

        Returns:
            iou: Tensor, average iou over the batch.
            update_op: Tensor op, update operation for the iou metric.
        """
        mask = tf.reshape(tf.one_hot(tf.squeeze(mask), depth=self.num_classes), [
            tf.shape(mask)[0], -1, self.num_classes])
        prediction = tf.reshape(
            prediction, shape=[tf.shape(prediction)[0], -1, self.num_classes])
        iou, update_op = tf.metrics.mean_iou(
            tf.argmax(prediction, 2), tf.argmax(mask, 2), self.num_classes)

        return iou, update_op

    @staticmethod
    def batch_norm(x, training, name):
        """
        Wrapper for batch normalization in tensorflow, updates moving batch statistics
        if training, uses trained parameters if inferring.
        Args:
            x: Tensor, the input to normalize.
            training: Boolean tensor, indicates if training or not.
            name: String, name of the op in the graph.

        Returns:
            x: Batch normalized input.
        """
        with tf.variable_scope(name):
            x = tf.cond(training, lambda: tf.contrib.layers.batch_norm(x, is_training=True, scope=name+'_batch_norm'),
                        lambda: tf.contrib.layers.batch_norm(x, is_training=False, scope=name+'_batch_norm', reuse=True))
        return x

    def conv_layer(self, x, training, filters, name):
        """
        Forms the atomic layer of the tiramisu, does three operation in sequence:
        batch normalization -> Relu -> 2D Convolution.
        Args:
            x: Tensor, input feature map.
            training: Bool Tensor, indicating whether training or not.
            filters: Integer, indicating the number of filters in the output feat. map.
            name: String, naming the op in the graph.

        Returns:
            x: Tensor, Result of applying batch norm -> Relu -> Convolution.
        """
        with tf.name_scope(name):
            x = self.batch_norm(x, training, name=name+'_bn')
            x = tf.nn.relu(x, name=name+'_relu')
            x = tf.layers.conv2d(x,
                                 filters=filters,
                                 kernel_size=[3, 3],
                                 strides=[1, 1],
                                 padding='SAME',
                                 dilation_rate=[1, 1],
                                 activation=None,
                                 kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                 name=name+'_conv3x3')
            x = tf.layers.dropout(x, rate=0.2, training=training, name=name+'_dropout')

        return x

    def dense_block(self, x, training, block_nb, name):
        """
        Forms the dense block of the Tiramisu to calculate features at a specified growth rate.
        Each conv layer in the dense block calculate growth_k feature maps, which are sequentially
        concatenated to build a larger final output.
        Args:
            x: Tensor, input to the Dense Block.
            training: Bool Tensor, indicating whether training or testing.
            block_nb: Int, identifying the block in the graph.
            name: String, identifying the layers in the graph.

        Returns:
            x: Tesnor, the output of the dense block.
        """
        dense_out = []
        with tf.name_scope(name):
            for i in range(self.layers_per_block[block_nb]):
                conv = self.conv_layer(x, training, self.growth_k, name=name+'_layer_'+str(i))
                x = tf.concat([conv, x], axis=3)
                dense_out.append(conv)

            x = tf.concat(dense_out, axis=3)

        return x

    def transition_down(self, x, training, filters, name):
        """
        Down-samples the input feature map by half using maxpooling.
        Args:
            x: Tensor, input to downsample.
            training: Bool tensor, indicating whether training or inferring.
            filters: Integer, indicating the number of output filters.
            name: String, identifying the ops in the graph.

        Returns:
            x: Tensor, result of downsampling.
        """
        with tf.name_scope(name):
            x = self.batch_norm(x, training, name=name+'_bn')
            x = tf.nn.relu(x, name=name+'relu')
            x = tf.layers.conv2d(x,
                                 filters=filters,
                                 kernel_size=[1, 1],
                                 strides=[1, 1],
                                 padding='SAME',
                                 dilation_rate=[1, 1],
                                 activation=None,
                                 kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                 name=name+'_conv1x1')
            x = tf.layers.dropout(x, rate=0.2, training=training, name=name+'_dropout')
            x = tf.nn.max_pool(x, [1, 4, 4, 1], [1, 2, 2, 1], padding='SAME', name=name+'_maxpool2x2')

        return x

    def transition_up(self, x, filters, name):
        """
        Up-samples the input feature maps using transpose convolutions.
        Args:
            x: Tensor, input feature map to upsample.
            filters: Integer, number of filters in the output.
            name: String, identifying the op in the graph.

        Returns:
            x: Tensor, result of up-sampling.
        """
        with tf.name_scope(name):
            x = tf.layers.conv2d_transpose(x,
                                           filters=filters,
                                           kernel_size=[3, 3],
                                           strides=[2, 2],
                                           padding='SAME',
                                           activation=None,
                                           kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                           name=name+'_trans_conv3x3')

        return x

    def model(self, x, training):
        """
        Defines the complete graph model for the Tiramisu based on the provided
        parameters.
        Args:
            x: Tensor, input image to segment.
            training: Bool Tesnor, indicating whether training or not.

        Returns:
            x: Tensor, raw unscaled logits of predicted segmentation.
        """
        concats = []
        with tf.variable_scope('encoder'):
            x = tf.layers.conv2d(x,
                                filters=48,
                                kernel_size=[3, 3],
                                strides=[1, 1],
                                padding='SAME',
                                dilation_rate=[1, 1],
                                activation=None,
                                kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                name='first_conv3x3')
            print("First Convolution Out: ", x.get_shape())
            for block_nb in range(0, self.nb_blocks):
                dense = self.dense_block(x, training, block_nb, 'down_dense_block_' + str(block_nb))

                if block_nb != self.nb_blocks - 1:
                    x = tf.concat([x, dense], axis=3, name='down_concat_' + str(block_nb))
                    concats.append(x)
                    x = self.transition_down(x, training, x.get_shape()[-1], 'trans_down_' + str(block_nb))
                    print("Downsample Out:", x.get_shape())

            x = dense
            print("Bottleneck Block: ", dense.get_shape())

        with tf.variable_scope('decoder'):
            for i, block_nb in enumerate(range(self.nb_blocks - 1, 0, -1)):
                x = self.transition_up(x, x.get_shape()[-1], 'trans_up_' + str(block_nb))
                x = tf.concat([x, concats[len(concats) - i - 1]], axis=3, name='up_concat_' + str(block_nb))
                print("Upsample after concat: ", x.get_shape())
                x = self.dense_block(x, training, block_nb, 'up_dense_block_' + str(block_nb))

        with tf.variable_scope('prediction'):
            x = tf.layers.conv2d(x,
                                filters=self.num_classes,
                                kernel_size=[1, 1],
                                strides=[1, 1],
                                padding='SAME',
                                dilation_rate=[1, 1],
                                activation=None,
                                kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                name='last_conv1x1')
            print("Mask Prediction: ", x.get_shape())

        return x

    def train(self, train_path, val_path, save_dir, batch_size, epochs, learning_rate):
        """
        Trains the Tiramisu on the specified training data and periodically validates
        on the validation data.

        Args:
            train_path: Directory where the training data is present.
            val_path: Directory where the validation data is present.
            save_dir: Directory where to save the model and training summaries.
            batch_size: Batch size to use for training.
            epochs: Number of epochs (complete passes over one dataset) to train for.
            learning_rate: Learning rate for the optimizer.
        Returns:
            None
        """

        train_image_path = os.path.join(train_path, 'images')
        train_mask_path = os.path.join(train_path, 'masks')
        val_image_path = os.path.join(val_path, 'images')
        val_mask_path = os.path.join(val_path, 'masks')

        assert os.path.exists(train_image_path), "No training image folder found"
        assert os.path.exists(train_mask_path), "No training mask folder found"
        assert os.path.exists(val_image_path), "No validation image folder found"
        assert os.path.exists(val_mask_path), "No validation mask folder found"

        train_image_paths, train_mask_paths = get_data_paths_list(train_image_path, train_mask_path)
        val_image_paths, val_mask_paths = get_data_paths_list(val_image_path, val_mask_path)

        assert len(train_image_paths) == len(train_mask_paths), "Number of images and masks dont match in train folder"
        assert len(val_image_paths) == len(val_mask_paths), "Number of images and masks dont match in validation folder"

        self.num_train_images = len(train_image_paths)
        self.num_val_images = len(val_image_paths)

        train_data, train_queue_init = utility.data_batch(
            train_image_paths, train_mask_paths, batch_size)
        train_image_tensor, train_mask_tensor = train_data

        eval_data, eval_queue_init = utility.data_batch(
            val_image_paths, val_mask_paths, batch_size)
        eval_image_tensor, eval_mask_tensor = eval_data

        image_ph = tf.placeholder(tf.float32, shape=[None, 256, 256, 3])
        mask_ph = tf.placeholder(tf.int32, shape=[None, 256, 256, 1])
        training = tf.placeholder(tf.bool, shape=[])

        if not self.logits:
            self.logits = self.model(image_ph, training)

        loss = tf.reduce_mean(self.xentropy_loss(self.logits, mask_ph))

        with tf.variable_scope("mean_iou_train"):
            iou, iou_update = self.calculate_iou(mask_ph, self.logits)

        optimizer = tf.train.AdamOptimizer(learning_rate)
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            opt = optimizer.minimize(loss)

        running_vars = tf.get_collection(
            tf.GraphKeys.LOCAL_VARIABLES, scope="mean_iou_train")

        reset_iou = tf.variables_initializer(var_list=running_vars)

        saver = tf.train.Saver(max_to_keep=20)
        with tf.Session() as sess:
            sess.run([tf.global_variables_initializer(),
                    tf.local_variables_initializer()])
            for epoch in range(epochs):
                writer = tf.summary.FileWriter(os.path.dirname(save_dir), sess.graph)
                sess.run([train_queue_init, eval_queue_init])
                total_train_cost, total_val_cost = 0, 0
                total_train_iou, total_val_iou = 0, 0
                for train_step in range(self.num_train_images // batch_size):
                    image_batch, mask_batch, _ = sess.run(
                        [train_image_tensor, train_mask_tensor, reset_iou])
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

                for val_step in range(self.num_val_images // batch_size):
                    image_batch, mask_batch, _ = sess.run(
                        [eval_image_tensor, eval_mask_tensor, reset_iou])
                    feed_dict = {image_ph: image_batch,
                                mask_ph: mask_batch,
                                training: True}
                    eval_cost, _ = sess.run(
                        [loss, iou_update], feed_dict=feed_dict)
                    eval_iou = sess.run(iou, feed_dict=feed_dict)
                    total_val_cost += eval_cost
                    total_val_iou += eval_iou

                print("Epoch: {0}, training loss: {1}, validation loss: {2}".format(epoch, 
                                    total_train_cost / train_step, total_val_cost / val_step))
                print("Epoch: {0}, training iou: {1}, val iou: {2}".format(epoch, 
                                    total_train_iou / train_step, total_val_iou / val_step))
                                    
                print("Saving model...")
                saver.save(sess, save_dir, global_step=epoch)

    def infer(self, image_dir, batch_size, ckpt, output_folder):
        """
        Uses a trained model file to get predictions on the specified images.
        Args:
            image_dir: Directory where the images are located.
            batch_size: Batch size to use while inferring (relevant if batch norm is used)
            ckpt: Name of the checkpoint file to use.
            output_folder: Folder where the predictions on the images shoudl be saved.
        """
        image_paths = [os.path.join(image_dir, x) for x in os.listdir(image_dir) if x.endswith('.png') or x.endswith('.jpg')]
        infer_data, infer_queue_init = utility.data_batch(
            image_paths, None, batch_size)
        image_ph = tf.placeholder(tf.float32, shape=[None, 256, 256, 3])
        training = tf.placeholder(tf.bool, shape=[])

        if not self.logits:
            self.logits = self.model(image_ph, training)

        mask = tf.squeeze(tf.argmax(self.logits, axis=3))

        saver = tf.train.Saver()
        with tf.Session() as sess:
            saver.restore(sess, ckpt)
            sess.run(infer_queue_init)
            for _ in range(len(image_paths) // batch_size):
                image = sess.run(infer_data)
                feed_dict = {
                    image_ph: image,
                    training: True
                }
                prediction = sess.run(mask, feed_dict)
                for j in range(prediction.shape[0]):
                    cv2.imwrite(os.path.join(output_folder, '{}.png'.format(j)), 255 * prediction[j, :, :])
