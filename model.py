import tensorflow as tf


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
        for i, block_nb in enumerate(range(self.nb_blocks - 1, 0, -1)):
            x = self.transition_up(x, x.get_shape()[-1], 'trans_up_' + str(block_nb))
            x = tf.concat([x, concats[len(concats) - i - 1]], axis=3, name='up_concat_' + str(block_nb))
            print("Upsample after concat: ", x.get_shape())
            x = self.dense_block(x, training, block_nb, 'up_dense_block_' + str(block_nb))

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
