import tensorflow as tf
from model import DenseTiramisu

class TestModel(tf.test.TestCase):

    def test_initialzation(self):
        tiramisu = DenseTiramisu(14, [2, 1], 2)
        self.assertIsNotNone(tiramisu)
        self.assertEqual(tiramisu.growth_k, 14)
        self.assertListEqual(tiramisu.layers_per_block, [2, 1])
        self.assertEqual(tiramisu.nb_blocks, 2)
        self.assertEqual(tiramisu.num_classes, 2)

    def test_xentropy_loss_length(self):
        tiramisu = DenseTiramisu(14, [2, 1], 2)
        logits = tf.constant([[[15.1], [14.1]], [[-12.1], [14.1]]])
        labels = tf.constant([[0.0], [1.0]])
        loss = tiramisu.xentropy_loss(logits, labels)

        with self.test_session():
            self.assertEqual(len(loss.eval()), 2)

    def test_xentropy_loss_correct(self):
        tiramisu = DenseTiramisu(14, [2, 1], 2)
        logits = tf.constant([[[15.1], [14.1]], [[-12.1], [14.1]]])
        labels = tf.constant([[0.0], [1.0]])
        loss = tiramisu.xentropy_loss(logits, labels)

        with self.test_session():
            self.assertGreaterEqual(loss.eval()[0][0], 0)
            self.assertEqual(loss.eval()[0][1], 0)

    def test_xentropy_loss_incorrect(self):
        tiramisu = DenseTiramisu(14, [2, 1], 2)
        logits = tf.constant([[[15.1], [14.1]], [[12.1], [-14.1]]])
        labels = tf.constant([[0.0], [1.0]])
        loss = tiramisu.xentropy_loss(logits, labels)

        with self.test_session():
            self.assertEqual(loss.eval()[0][0], 0)
            self.assertGreater(loss.eval()[0][1], 0)
