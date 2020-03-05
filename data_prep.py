from imports import *

# Image Normalization
def image_normalization(train, test):
    train_mean = np.mean(train, axis=(0,1,2))
    train_std = np.std(train, axis=(0,1,2))
    train_n = (train - train_mean) / train_std
    test_n = (test - train_mean) / train_std
    return train_n, test_n

# For float sequence of images
def float_tffeature(value) -> tf.train.Feature:
    if not isinstance(value, list):
        value = [value]
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))

# For int sequence of labels
def int_tffeature(value) -> tf.train.Feature:
    if not isinstance(value, list):
        value = [value]
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))

# For returning a dict value sequences with images and their labels
def numpy_tfexample(x, y=None) -> tf.train.Example:
    if y is None:
        feat_dict = {'image': float_tffeature(x.tolist())}
    else:
        feat_dict = {'image': float_tffeature(x.tolist()), 'label': int_tffeature(y)}
    return tf.train.Example(features=tf.train.Features(feature=feat_dict))

# For serializing the complete dataset into sequences of features and storing it
def numpy_tfrecord(output_fn, X, y=None, overwrite = False):
    n = X.shape[0]
    X_reshape = X.reshape(n, -1)

    if overwrite or not tf.io.gfile.exists(output_fn):
        with tf.io.TFRecordWriter(output_fn) as record_writer:
            for i in range(n):
                example = numpy_tfexample(X_reshape[i]) if y is None else numpy_tfexample(X_reshape[i], y[i])
                record_writer.write(example.SerializeToString())
    else:
        tf.logging.info('Output file already exists. Skipping.')


# Function for parsing every sequence linearly
def tfexample_numpy_image_parser(tfexample: tf.train.Example, h: int, w: int, c: int = 3, dtype=tf.float32) -> Tuple[
    tf.Tensor, tf.Tensor]:
    feat_dict = {'image': tf.io.FixedLenFeature([h * w * c], dtype),
                 'label': tf.io.FixedLenFeature([], tf.int64)}
    feat = tf.io.parse_single_example(tfexample, features=feat_dict)
    x, y = feat['image'], feat['label']
    x = tf.reshape(x, [h, w, c])
    return x, y

def tfrecord_ds(file_pattern: str, parser, batch_size: int, training: bool = True,
                shuffle_buf_sz: int = 50000, n_cores: int = 4) -> tf.data.Dataset:

    # Data extraction
    files = tf.data.Dataset.list_files(file_pattern)

    # Parallelized Data Extraction
    dataset = files.apply(tf.data.experimental.parallel_interleave(tf.data.TFRecordDataset,
                                                                   cycle_length=n_cores,
                                                                   sloppy=True))

    # For training data, randomize the order of sequences and repeat
    if training:
        dataset = dataset.shuffle(shuffle_buf_sz)
        dataset = dataset.repeat()

    # Parallelized Data Transformation
    dataset = dataset.apply(tf.data.experimental.map_and_batch(map_func=parser, batch_size=batch_size,
                                                               num_parallel_batches=n_cores,
                                                               drop_remainder=True))
    # Pipelining the data to decrease idle time
    dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)
    return dataset
