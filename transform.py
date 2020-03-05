from imports import *

# Function for implementing cutout, replacing the a size of 8x8 with a mask of random values
def replace_slice(input_: tf.Tensor, replacement, begin) -> tf.Tensor:
    inp_shape = tf.shape(input_)
    size = tf.shape(replacement)
    padding = tf.stack([begin, inp_shape - (begin + size)], axis=1)
    replacement_pad = tf.pad(replacement, padding)
    mask = tf.pad(tf.ones_like(replacement, dtype=tf.bool), padding)
    return tf.where(mask, replacement_pad, input_)

# Cutout image augmentation for sequence of data
def cutout(x: tf.Tensor, h: int, w: int, c: int = 3) -> tf.Tensor:
    shape = tf.shape(x)
    x0 = tf.random.uniform([], 0, shape[0] + 1 - h, dtype=tf.int32)
    y0 = tf.random.uniform([], 0, shape[1] + 1 - w, dtype=tf.int32)
    x = replace_slice(x, tf.zeros([h, w, c]), [x0, y0, 0])
    return x

# Horizontal flip image augmentation for sequence of data
def horizontal_flip(x: tf.Tensor) -> tf.Tensor:
    x = tf.image.random_flip_left_right(x)
    return x

# Pad 'reflect' and crop image augmentation for sequence of data
def random_pad_crop(x: tf.Tensor, pad_size: int) -> tf.Tensor:
    shape = tf.shape(x)
    x = tf.pad(x, [[pad_size, pad_size], [pad_size, pad_size], [0, 0]], mode='reflect')
    x = tf.image.random_crop(x, [shape[0], shape[1], 3])
    return x
