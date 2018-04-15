import os
import argparse
from train import TrainEval

parser = argparse.ArgumentParser()
parser.add_argument("--train_images", default="data/training/images", help="Directory for training images")
parser.add_argument("--train_masks", default="data/training/masks", help="Directory for training masks")
parser.add_argument("--val_images", default="data/validation/images", help="Directory for validation images")
parser.add_argument("--val_masks", default="data/validation/masks", help="Directory for validation masks")
parser.add_argument("--ckpt_dir", default="models/model.ckpt", help="Directory for storing model checkpoints")
parser.add_argument("--layers_per_block", default="2,2", help="Number of layers in dense blocks")
parser.add_argument("--batch_size", default=2, help="Batch size for use in training")
parser.add_argument("--epochs", default=10, help="Number of epochs for training")
parser.add_argument("--num_threads", default=2, help="Number of threads to use for data input pipeline")
parser.add_argument("--growth_k", default=16, help="Growth rate for Tiramisu")
parser.add_argument("--num_classes",   default=2, help="Number of classes")
parser.add_argument("--learning_rate", default=1e-4, help="Learning rate for optimizer")

def get_data_paths_list(image_folder, mask_folder):
    """Returns lists of paths to each image and mask."""

    image_paths = [os.path.join(image_folder, x) for x in os.listdir(image_folder) if x.endswith(".png")]
    mask_paths = [os.path.join(mask_folder, os.path.basename(x)) for x in image_paths]

    return image_paths, mask_paths


if __name__ == "__main__":
    FLAGS = parser.parse_args()
    layers_per_block = [int(x) for x in FLAGS.layers_per_block.split(",")]
    print(layers_per_block)

    image_paths, mask_paths = get_data_paths_list(FLAGS.train_images, FLAGS.train_masks)
    eval_image_paths, eval_mask_paths = get_data_paths_list(FLAGS.val_images, FLAGS.val_masks)

    train_eval = TrainEval(image_paths, mask_paths, eval_image_paths, eval_mask_paths, FLAGS.ckpt_dir,
                                    FLAGS.num_classes)

    train_eval.train_eval(FLAGS.batch_size, FLAGS.growth_k, layers_per_block, FLAGS.epochs, FLAGS.learning_rate)

