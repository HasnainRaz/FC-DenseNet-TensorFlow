import tensorflow as tf
import numpy as np


def xentropy_loss(logits, labels, batch_size):
    """Calculates the cross-entropy loss for predictions and labels."""
    labels = tf.cast(labels, tf.int32)
    logits = tf.reshape(logits, [tf.shape(logits)[0], -1, 2])
    labels = tf.reshape(labels, [tf.shape(labels)[0], -1])
    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
        logits=logits, labels=labels, name="loss")

    return loss


def focal_loss(logits, labels, gamma=0.2, alpha=0.01):
    epsilon = 1.e-9
    labels = tf.cast(labels, tf.int32)
    logits = tf.reshape(tf.nn.softmax(logits), [-1, 2])
    labels = tf.reshape(tf.one_hot(tf.squeeze(labels), depth=2), [-1, 2])

    model_out = tf.add(logits, epsilon)
    ce = tf.multiply(labels, -tf.log(model_out))
    weight = tf.multiply(labels, tf.pow(tf.subtract(1., model_out), gamma))
    fl = tf.multiply(alpha, tf.multiply(weight, ce))
    reduced_fl = tf.reduce_max(fl, axis=1)

    return reduced_fl


class DiscriminativeLoss(object):

    def __init__(self, delta_v, delta_d, norm_type):
        self.delta_v = delta_v
        self.delta_d = delta_d
        self.norm_type = norm_type

    def loop_body_means(self, i, nb_objects, pred_masked, mask_repeated, means):
        pred_masked_sample = pred_masked[i, :, :nb_objects[i]]
        gt_expanded_sample = mask_repeated[i, :, :nb_objects[i]]

        mean_sample = tf.reduce_sum(pred_masked_sample, axis=0) / tf.reduce_sum(gt_expanded_sample, axis=0)
        means.write(i, mean_sample)
        tf.assign_add(i, 1)

        return i, nb_objects, pred_masked, mask_repeated, means

    def calculate_loss(self, logits, mask, n_objects):
        alpha, beta = 1.0, 1.0
        gamma = 0.001

        logits_batch, logits_height, logits_width, logits_channels = logits.shape()
        truth_batch, truth_height, truth_width, truth_channels = mask.shape()

        logits = tf.reshape(logits, [logits_batch, logits_height * logits_width, logits_channels])
        mask = tf.reshape(mask, [truth_batch, truth_height * truth_width, truth_channels])

        cluster_means = self.calculate_means(logits, mask, n_objects)

        var_term = self.calculate_variance_term(logits, mask, cluster_means, n_objects, self.delta_v)
        dist_term = self.calculate_distance_term(logits, mask, cluster_means, n_objects, self.delta_d)
        reg_term = self.reg_term(cluster_means, n_objects)

        loss = alpha * var_term + beta * dist_term + gamma * reg_term

        return loss

    def calculate_means(self, logits, mask, nb_objects):
        batch, nb_loc, nb_filters = logits.shape()
        nb_instances = mask.shape()[2]

        logits_repeated = tf.tile(tf.expand_dims(logits, axis=2), (1, 1, nb_instances, 1))
        mask_repeated = tf.expand_dims(mask, axis=3)

        pred_masked = tf.mutiply(logits_repeated, mask_repeated)

        i = tf.constant(0)
        means = tf.TensorArray(tf.float32, [batch])
        _, _, _, _, means_final = tf.while_loop(tf.less(i, tf.shape(logits)[0]), self.loop_body_means,
                                                (i, nb_objects, pred_masked, mask_repeated, means))

        means_stacked = tf.stack(means)

        return means_stacked

    def loop_var(self, i, gt, var_term, var, n_objects):
        var_sample = var[i, :, :n_objects]
        gt_sample = gt[i, :, :n_objects]
        var_term.write(i, tf.reduce_sum(var_sample) / tf.reduce_sum(gt_sample))
        tf.assign_add(i, 1)

        return i, gt, var_term, var, n_objects

    def calculate_variance_term(self, pred, gt, means, n_objects, delta_v):
        bs, n_loc, n_filters = tf.shape(pred)
        n_instances = tf.shape(gt)[3]

        means = tf.tile(tf.expand_dims(means, axis=1), (1, n_loc, 1, 1))
        pred = tf.tile(tf.expand_dims(pred, axis=2), (1, 1, n_instances, 1))

        gt = tf.tile(tf.expand_dims(gt, axis=3), (1, 1, 1, n_filters))

        var = tf.pow(tf.clip_by_value(tf.norm((pred - means), axis=3) - delta_v, min=0.0, max=np.infty), 2) * gt[:, :,
                                                                                                              :, 0]

        var_term = tf.TensorArray(tf.float32, size=[bs])
        i = tf.constant(0)
        _, var_term_modded, n_objects = tf.while_loop(tf.less(i, bs), self.loop_var, (i, gt, var_term, var, n_objects))

        v_term_stacked = tf.stack(var_term_modded)

        var_term_val = tf.reduce_sum(v_term_stacked) / bs

        return var_term_val

    def loop_dist(self, i, means, dist_term, n_objects, delta_d):
        n_objects_sample = n_object[i]

        mean_sample = means[i, :n_objects_sample, :]
        means_1 = tf.tile(tf.expand_dims(mean_sample, axis=1), (1, n_objects_sample, 1))
        means_2 = tf.transpose(means_1, [1, 0, 2])

        diff = means_1 - means_2

        norm = tf.norm(diff, axis=2)

        margin = 2 * delta_d * (1.0 - tf.eye(n_objects_sample))
        dist_term_sample = tf.reduce_sum(tf.pow(tf.clip_by_value((margin - norm), min=0.0, max=np.infty), 2))
        dist_term_sample = dist_term_sample / (n_objects_sample * (n_objects_sample - 1))

        dist_term.write(dist_term_sample, i)
        tf.assign_add(i, 1)

        return i, means, dist_term, n_objects, delta_d

    def calculate_distance_term(self, means, n_objects, delta_d):
        bs, n_instances, n_filters = tf.shape(means)

        dist_term = tf.TensorArray(tf.float32, size=[bs])
        i = tf.constant(0)
        _, _, dist_term_modded, _, _ = tf.while_loop(tf.less(i, bs), self.loop_dist,
                                                     (i, means, dist_term, n_objects, delta_d))

        dist_term_val = tf.reduce_sum(tf.stack(dist_term_modded)) / bs

        return dist_term_val

    def loop_reg(self, i, means, reg_term, n_objects):
        mean_sample = means[i, :n_objects[i], :]
        norm = tf.norm(mean_sample, axis=1)
        reg_term.write(tf.reduce_mean(norm), i)

        tf.assign_add(i, 1)

        return i, means, reg_term, n_objects

    def reg_term(self, means, n_objects):
        bs, n_instances, n_filters = tf.shape(means)

        reg_term = tf.TensorArray(tf.float32, size=[bs])
        i = tf.constant(0)
        _, _, reg_term_vals, _ = tf.while_loop(tf.less(i, bs), self.loop_reg, (i, means, reg_term, n_objects))

        reg_term_vals = tf.reduce_sum(tf.stack(reg_term_vals)) / bs

        return reg_term_vals
