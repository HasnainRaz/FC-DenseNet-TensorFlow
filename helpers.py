import os

def get_data_paths_list(image_folder, mask_folder):
    """Returns lists of paths to each image and mask."""

    image_paths = [os.path.join(image_folder, x) for x in os.listdir(
        image_folder) if x.endswith(".png")]
    mask_paths = [os.path.join(mask_folder, os.path.basename(x))
                  for x in image_paths]

    return image_paths, mask_paths