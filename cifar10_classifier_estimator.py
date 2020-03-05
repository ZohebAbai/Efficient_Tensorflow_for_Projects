# Complex but complete system importing several files.
from imports import *
import io_dir, data_prep, layers, model_arch, train, optimizers, transform

# Hyperparameters
BATCH_SIZE = 512
MOMENTUM = 0.9
LEARNING_RATE = 0.4
WEIGHT_DECAY = 5e-4
EPOCHS = 24
WARMUP = 5
ROOT_DIR = '.'

# Function to classify cifar10 dataset
def CIFAR10_classifier_estimator(_):

    # Define two separate directories for storing data and train/eval related files
    data_dir, work_dir = io_dir.get_dirs(ROOT_DIR)

    # Downloading datasets
    (X_train, y_train), (X_test, y_test) = tf.keras.datasets.cifar10.load_data()
    len_train, len_test = len(X_train), len(X_test)
    img_size = X_train.shape[1]
    n_classes = y_train.max() + 1

    # Image Normalization the data
    X_train, X_test = data_prep.image_normalization(X_train, X_test)

    # Create the directories to store the tfrecords data
    train_fn = os.path.join(data_dir, "train.tfrec")
    test_fn = os.path.join(data_dir, "test.tfrec")

    # Call the function for test and train dataset
    data_prep.numpy_tfrecord(train_fn, X_train, y_train)
    data_prep.numpy_tfrecord(test_fn, X_test, y_test)

    #Image Augmentation techniques
    #Random Crop of 32 with padding of 4px
    #Horizontal Flip
    #CutOut of 8
    # parser dunction for training dataset
    def parser_train(tfexample):
        x, y = data_prep.tfexample_numpy_image_parser(tfexample, img_size, img_size)
        x = transform.random_pad_crop(x, 4)
        x = transform.horizontal_flip(x)
        x = transform.cutout(x, 8, 8)
        return x, y

    # parser function for test dataset
    parser_test = lambda x: data_prep.tfexample_numpy_image_parser(x, img_size, img_size)

    #Input Data Pipeline
    #Extract: Read data from persistent storage
    #Transform: Use CPU cores to parse and perform preprocessing operations on the data
    #Load: Load the transformed data onto GPU for executing ML model
    train_input_func = lambda : data_prep.tfrecord_ds(train_fn, parser_train, batch_size= BATCH_SIZE, training=True)
    eval_input_func = lambda : data_prep.tfrecord_ds(test_fn, parser_test, batch_size=BATCH_SIZE, training=False)

    # Gradient Update and Final estimator specs required to pass to an Estimator
    # defining required parameters
    steps_per_epoch = len_train // BATCH_SIZE
    total_steps = steps_per_epoch * EPOCHS
    warmup_steps = steps_per_epoch * WARMUP

    # Defining one cycle learning rate manager
    lr_func = train.one_cycle_lr(LEARNING_RATE/BATCH_SIZE, total_steps, warmup_steps, train.linear_decay())

    # optimizer function with weight decay
    opt_func = train.sgd_optimizer(lr_func, mom = MOMENTUM, wd=WEIGHT_DECAY*BATCH_SIZE)

    # Model specs to be passed to tf.estimator
    model_func = train.get_model_func(model_arch.Resnet9, opt_func, work_dir,
                                        reduction = tf.losses.Reduction.SUM)

    # Create a estimator with model_fn
    image_classifier = tf.estimator.Estimator(model_fn = model_func, model_dir = work_dir)

    # Finally, train and evaluate the model after each epoch
    image_classifier.train(input_fn = train_input_func, steps= total_steps)
    metrics = image_classifier.evaluate(input_fn = eval_input_func, steps= len_test // 1000)

if __name__ == '__main__':
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.INFO)
#Run the code
tf.compat.v1.app.run(CIFAR10_classifier_estimator)
