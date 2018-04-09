import tensorflow as tf


def calculate_iou(mask, prediction, num_classes=2):
    """Calculates mean IoU, the default eval metric of segmentation."""
    mask = tf.reshape(tf.one_hot(tf.squeeze(mask), depth=num_classes), [
                      tf.shape(mask)[0], -1, num_classes])
    prediction = tf.reshape(
        prediction, shape=[tf.shape(prediction)[0], -1, num_classes])
    iou, update_op = tf.metrics.mean_iou(
        tf.argmax(prediction, 2), tf.argmax(mask, 2), num_classes)

    return iou, update_op
