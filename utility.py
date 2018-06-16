import tensorflow as tf


def _parse_data_infer(image_paths):
    image_content = tf.read_file(image_paths)
    images = tf.image.decode_png(image_content, channels=3)

    return images


def _normalize_data_infer(image):
    image = tf.cast(image, tf.float32)
    image = tf.divide(image, 255.0)

    return image


def _resize_data_infer(image):
    image = tf.image.resize_images(image, [256, 256])

    return image


def _normalize_data(image, mask):
    """Normalizes data in between 0-1"""
    image = tf.cast(image, tf.float32)
    image = image / 255.0

    mask = tf.cast(mask, tf.float32)
    mask = mask / 255.0

    return image, mask


def _resize_data(image, mask):
    """Resizes images to smaller dimensions."""
    image = tf.image.resize_images(image, [256, 256])
    mask = tf.image.resize_images(mask, [256, 256])

    return image, mask


def _parse_data(image_paths, mask_paths):
    """Reads image and mask files"""
    image_content = tf.read_file(image_paths)
    images = tf.image.decode_png(image_content, channels=3)

    mask_content = tf.read_file(mask_paths)
    masks = tf.image.decode_png(mask_content, channels=1)

    return images, masks


def data_batch(image_paths, mask_paths, batch_size=4, augment=False, num_threads=2):
    """Reads data, normalizes it, shuffles it, then batches it, returns a
       the next element in dataset op and the dataset initializer op.
       Inputs:
        image_paths: A list of paths to individual images
        mask_paths: A list of paths to individual mask images
        batch_size: Number of images/masks in each batch returned
        num_threads: Number of parallel calls to make
       Returns:
        next_element: A tensor with shape [2], where next_element[0]
                      is image batch, next_element[1] is the corresponding
                      mask batch
        init_op: Data initializer op, needs to be executed in a session
                 for the data queue to be filled up and the next_element op
                 to yield batches"""

    # Convert lists of paths to tensors for tensorflow

    return tf.constant(0), tf.constant(0)
    
    images_name_tensor = tf.constant(image_paths)

    if mask_paths:
        mask_name_tensor = tf.constant(mask_paths)
        data = tf.data.Dataset.from_tensor_slices(
            (images_name_tensor, mask_name_tensor))
        data = data.map(
            _parse_data, num_parallel_calls=num_threads).prefetch(30)
        data = data.map(
            _resize_data, num_parallel_calls=num_threads).prefetch(30)
        data = data.map(_normalize_data,
                        num_parallel_calls=num_threads).prefetch(30)
    else:
        data = tf.data.Dataset.from_tensor_slices((images_name_tensor))
        data = data.map(_parse_data_infer,
                        num_parallel_calls=num_threads).prefetch(30)
        data = data.map(_resize_data_infer,
                        num_parallel_calls=num_threads).prefetch(30)
        data = data.map(_normalize_data_infer,
                        num_parallel_calls=num_threads).prefetch(30)

    # Batch the data
    data = data.batch(batch_size)

    data = data.shuffle(30)

    # Create iterator
    iterator = tf.data.Iterator.from_structure(
        data.output_types, data.output_shapes)

    # Next element Op
    next_element = iterator.get_next()

    # Data set init. op
    init_op = iterator.make_initializer(data)

    return next_element, init_op
