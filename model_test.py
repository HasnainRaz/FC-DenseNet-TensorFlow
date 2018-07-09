import tensorflow as tf
from model import DenseTiramisu
import numpy as np


class TestModel(tf.test.TestCase):

    def setUp(self):
        self.tiramisu = DenseTiramisu(14, [2, 1], 2)

    def test_initialzation(self):
        tiramisu = DenseTiramisu(14, [2, 1], 2)
        self.assertIsNotNone(tiramisu)
        self.assertEqual(tiramisu.growth_k, 14)
        self.assertListEqual(tiramisu.layers_per_block, [2, 1])
        self.assertEqual(tiramisu.nb_blocks, 2)
        self.assertEqual(tiramisu.num_classes, 2)

    def test_xentropy_loss_length(self):
        logits = tf.constant([[[15.1], [14.1]], [[-12.1], [14.1]]])
        labels = tf.constant([[0.0], [1.0]])
        loss = self.tiramisu.xentropy_loss(logits, labels)

        with self.test_session():
            self.assertEqual(len(loss.eval()), 2)

    def test_xentropy_loss_correct(self):
        logits = tf.constant([[[15.1], [14.1]], [[-12.1], [14.1]]])
        labels = tf.constant([[0.0], [1.0]])
        loss = self.tiramisu.xentropy_loss(logits, labels)

        with self.test_session():
            self.assertLessEqual(loss.eval()[0], 0.5)
            self.assertLessEqual(loss.eval()[1], 0.5)

    def test_xentropy_loss_incorrect(self):
        logits = tf.constant([[[15.1], [14.1]], [[12.1], [-14.1]]])
        labels = tf.constant([[0.0], [1.0]])
        loss = self.tiramisu.xentropy_loss(logits, labels)

        with self.test_session():
            self.assertLessEqual(loss.eval()[0], 0.5)
            self.assertGreaterEqual(loss.eval()[1], 0.5)
            
    def test_iou_all_correct(self):
        logits = tf.constant([[[15.0], [1.0]], [[-1], [-10]]])
        labels = tf.constant([[0], [0]])
        iou, update_op = self.tiramisu.calculate_iou(labels, logits)

        with self.test_session() as sess:
            sess.run(tf.local_variables_initializer())
            self.assertEqual(iou.eval(), 0.0)
            update_op.eval()
            self.assertEqual(iou.eval(), 1.0)

    def test_iou_all_wrong(self):
        logits = tf.constant([[[15.0], [1.0]], [[-1], [-10]]])
        labels = tf.constant([[1], [1]])
        iou, update_op = self.tiramisu.calculate_iou(labels, logits)

        with self.test_session() as sess:
            sess.run(tf.local_variables_initializer())
            self.assertEqual(iou.eval(), 0.0)
            update_op.eval()
            self.assertEqual(iou.eval(), 0.0)


    def test_batch_norm(self):
        rand_tensor = tf.random_normal(shape=[2, 100, 100, 3], mean=127)
        training = tf.constant(True)
        normed_tensor = self.tiramisu.batch_norm(rand_tensor, training, 'test_bn')

        with self.test_session() as sess:
            sess.run(tf.global_variables_initializer())
            tensor, norm_tensor = sess.run([rand_tensor, normed_tensor])
            mean_tensor, mean_norm_tensor = np.mean(tensor), np.mean(norm_tensor)
            self.assertIsNotNone(tensor)
            self.assertIsNotNone(norm_tensor)
            self.assertShapeEqual(norm_tensor, rand_tensor)
            self.assertNotEqual(mean_tensor, mean_norm_tensor)
    
    def test_conv_layer_out_dims(self):
        rand_tensor = tf.random_normal(shape=[2, 100, 100, 3], mean=127) 
        training = tf.constant(True)
        conv_out = self.tiramisu.conv_layer(rand_tensor,
                                                 training,
                                                 56,
                                                 'test_conv')
        with self.test_session() as sess:
            sess.run(tf.global_variables_initializer())
            self.assertListEqual([2, 100, 100, 56], list(conv_out.eval().shape))
            
    def test_dense_block(self):
        rand_tensor = tf.random_normal(shape=[2, 100, 100, 3], mean=127)
        training = tf.constant(True)
        dense_out = self.tiramisu.dense_block(rand_tensor,
                                              training,
                                              0,
                                              'test_block')
        conv_layers = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='test_block_layer_0_conv3x3')
        with self.test_session() as sess:
            sess.run(tf.global_variables_initializer())
            self.assertListEqual([2, 100, 100, 28], list(dense_out.eval().shape))
            self.assertEqual(len(conv_layers), 2)

    def test_upsample_layer(self):
        rand_tensor = tf.random_normal(shape=[2, 100, 100, 3], mean=127)
        upsampled_example = self.tiramisu.transition_up(rand_tensor, 20, 'upsample')
        with self.test_session() as sess:
            sess.run(tf.global_variables_initializer())
            result = sess.run(upsampled_example)
            self.assertListEqual([2, 200, 200, 20], list(result.shape))

    def test_downsample_layer(self):
        rand_tensor = tf.random_normal(shape=[2, 100, 100, 3], mean=100)
        training = tf.constant(True)
        downsampled_tensor = self.tiramisu.transition_down(rand_tensor, training, 32, 'trans_down')
        with self.test_session() as sess:
            sess.run(tf.global_variables_initializer())
            out_tensor = sess.run(downsampled_tensor)
            self.assertListEqual([2, 50, 50, 32], list(out_tensor.shape))
    
    def test_model(self):
        tf.reset_default_graph()
        rand_tensor = tf.random_normal(shape=[2, 100, 100, 3], mean=127)
        training = tf.constant(True)
        logits = self.tiramisu.model(rand_tensor, training)
        encoder_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='encoder')
        decoder_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='decoder')
        predictions = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='prediction')

        with self.test_session() as sess:
            sess.run(tf.global_variables_initializer())
            logits_values = sess.run(logits)
        self.assertIsNotNone(encoder_vars)
        self.assertIsNotNone(decoder_vars)
        self.assertIsNotNone(predictions)
        self.assertListEqual([2, 100, 100, 2], list(logits_values.shape))

    def test_model_connectivity(self):
        tf.reset_default_graph()
        image_ph = tf.placeholder(tf.float32, shape=[2, 100, 100, 3])
        labels = tf.ones(shape=[2, 100, 100, 1])
        training = tf.placeholder(tf.bool)
        logits = self.tiramisu.model(image_ph, training)
        loss = self.tiramisu.xentropy_loss(logits, labels)
        opt = tf.train.AdamOptimizer(learning_rate=0.001).minimize(loss)
        input_img = np.random.randint(0, 256, size=[2, 100, 100, 3])
        with self.test_session() as sess:
            sess.run(tf.global_variables_initializer())
            before = sess.run(tf.trainable_variables())
            _ = sess.run(opt, feed_dict={image_ph: input_img,
                                         training: True})
            after = sess.run(tf.trainable_variables())
            # Check that none of the variables are equal before and after 
            # optimization
            for b, a in zip(before, after):
                assertion = (b != a).any()
                self.assertTrue(assertion)