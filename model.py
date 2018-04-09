import tensorflow as tf
import utility

class DenseTiramisu(object):

    def __init__(self, growth_k, layers_per_block, num_classes):
        self.growth_k = growth_k
        self.layers_per_block = layers_per_block
        self.nb_blocks = len(layers_per_block)
        self.num_classes = num_classes

    def batch_norm(self, x, training, name):
        with tf.variable_scope(name):
            x = tf.cond(training, lambda: tf.contrib.layers.batch_norm(x, is_training=True, scope=name+'_batch_norm'),
                        lambda: tf.contrib.layers.batch_norm(x, is_training=False, scope=name+'_batch_norm', reuse=True))
        return x

    def conv_layer(self, x, training, filters, name):
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
        dense_out = []
        with tf.name_scope(name):
            for i in range(self.layers_per_block[block_nb]):
                conv = self.conv_layer(x, training, self.growth_k, name=name+'_layer_'+str(i))
                x = tf.concat([conv, x], axis=3)
                dense_out.append(conv)

            x = tf.concat(dense_out, axis=3)

        return x

    def transition_down(self, x, training, filters, name):
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
        print(x.get_shape())
        print("Building downsample path...")
        for block_nb in range(0, self.nb_blocks):
            dense = self.dense_block(x, training, block_nb, 'down_dense_block_' + str(block_nb))

            if block_nb != self.nb_blocks - 1:
                x = tf.concat([x, dense], axis=3, name='down_concat_' + str(block_nb))
                print(x.get_shape())
                concats.append(x)
                x = self.transition_down(x, training, x.get_shape()[-1], 'trans_down_' + str(block_nb))

        print(dense.get_shape())
        print("Building upsample path...")
        for i, block_nb in enumerate(range(self.nb_blocks - 1, 0, -1)):
            x = self.transition_up(x, x.get_shape()[-1], 'trans_up_' + str(block_nb))
            x = tf.concat([x, concats[len(concats) - i - 1]], axis=3, name='up_concat_' + str(block_nb))
            print(x.get_shape())
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
        print(x.get_shape())

        return x
