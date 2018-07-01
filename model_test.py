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

    