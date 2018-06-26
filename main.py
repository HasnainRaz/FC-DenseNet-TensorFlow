import argparse
from train import TrainEval
from helpers import get_data_paths_list

parser = argparse.ArgumentParser()
parser.add_argument("--train_images", default="data/training/images",
                    help="Directory for training images")
parser.add_argument("--train_masks", default="data/training/masks",
                    help="Directory for training masks")
parser.add_argument("--val_images", default="data/validation/images",
                    help="Directory for validation images")
parser.add_argument("--val_masks", default="data/validation/masks",
                    help="Directory for validation masks")
parser.add_argument("--ckpt_dir", default="models/model.ckpt",
                    help="Directory for storing model checkpoints")
parser.add_argument("--layers_per_block", default="2,3,3",
                    help="Number of layers in dense blocks")
parser.add_argument("--batch_size", default=8,
                    help="Batch size for use in training", type=int)
parser.add_argument("--epochs", default=20,
                    help="Number of epochs for training", type=int)
parser.add_argument("--num_threads", default=2,
                    help="Number of threads to use for data input pipeline", type=int)
parser.add_argument("--growth_k", default=16, help="Growth rate for Tiramisu", type=int)
parser.add_argument("--num_classes",   default=2, help="Number of classes", type=int)
parser.add_argument("--learning_rate", default=1e-4,
                    help="Learning rate for optimizer", type=float)


def main():
    FLAGS = parser.parse_args()
    layers_per_block = [int(x) for x in FLAGS.layers_per_block.split(",")]

    try:
        image_paths, mask_paths = get_data_paths_list(
            FLAGS.train_images, FLAGS.train_masks)
        eval_image_paths, eval_mask_paths = get_data_paths_list(
            FLAGS.val_images, FLAGS.val_masks)
    except FileNotFoundError:
        print("No images found the directory specified directory")

    assert len(image_paths) == len(mask_paths), "Number of train images and masks found is different"
    assert len(eval_image_paths) == len(eval_mask_paths), "Number of validation images and masks found is different"
    assert len(image_paths // FLAGS.batch_size) > 0, "Number of training images less than batch size"
    assert len(eval_image_paths // FLAGS.batch_size) > 0, "Number of validation images less than batch size"

    train_eval = TrainEval(image_paths, mask_paths, eval_image_paths, eval_mask_paths, FLAGS.ckpt_dir,
                           FLAGS.num_classes)

    train_eval.train_eval(FLAGS.batch_size, FLAGS.growth_k,
                          layers_per_block, FLAGS.epochs, FLAGS.learning_rate)


if __name__ == "__main__":
    main()