# Simple system in single file
import os
import sys
import time
import tensorflow as tf
import numpy as np

# Hyperparameters
_NUM_CLASSES = 10
_MODEL_DIR = '.'
_NUM_CHANNELS = 1
_IMG_SIZE = 28
_LEARNING_RATE = 0.05
_NUM_EPOCHS = 3
_BATCH_SIZE = 2048

# Simple Model
class Model(object):
    def __call__(self, inputs):
        net = tf.layers.conv2d(inputs, 32, [5, 5], activation=tf.nn.relu, name='conv1')
        net = tf.layers.max_pooling2d(net, [2, 2], 2, name='pool1')
        net = tf.layers.conv2d(net, 64, [5, 5], activation=tf.nn.relu, name='conv2')
        net = tf.layers.max_pooling2d(net, [2, 2], 2, name='pool2')
        net = tf.layers.flatten(net)
        logits = tf.layers.dense(net, _NUM_CLASSES, activation=None, name='fc1')
        return logits

# Model Func
def model_fn(features, labels, mode):

    model = Model()
    global_step=tf.train.get_global_step()

    images = tf.reshape(features, [-1, _IMG_SIZE, _IMG_SIZE,
                        _NUM_CHANNELS])


    logits = model(images)
    predicted_logit = tf.argmax(input=logits, axis=1,
                                output_type=tf.int32)
    probabilities = tf.nn.softmax(logits)

    print(tf.compat.v1.trainable_variables())
    #PREDICT
    predictions = {
      "predicted_logit": predicted_logit,
      "probabilities": probabilities
    }
    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)
    with tf.name_scope('loss'):
        cross_entropy = tf.losses.sparse_softmax_cross_entropy(
            labels=labels, logits=logits, scope='loss')
        tf.summary.scalar('loss', cross_entropy)
    with tf.name_scope('accuracy'):
        accuracy = tf.metrics.accuracy(
            labels=labels, predictions=predicted_logit, name='acc')
        tf.summary.scalar('accuracy', accuracy[1])
    #EVAL
    if mode == tf.estimator.ModeKeys.EVAL:
        return tf.estimator.EstimatorSpec(
            mode=mode,
            loss=cross_entropy,
            eval_metric_ops={'accuracy/accuracy': accuracy},
            evaluation_hooks=None)


    # Create a SGR optimizer
    optimizer = tf.train.GradientDescentOptimizer(
                learning_rate=_LEARNING_RATE)
    train_op = optimizer.minimize(
                cross_entropy,global_step=global_step)

    # Create a hook to print acc, loss & global step every 100 iter.
    train_hook_list= []
    train_tensors_log = {'accuracy': accuracy[1],
                         'loss': cross_entropy,
                         'global_step': global_step}
    train_hook_list.append(tf.train.LoggingTensorHook(
        tensors=train_tensors_log, every_n_iter=100))

    if mode == tf.estimator.ModeKeys.TRAIN:
        return tf.estimator.EstimatorSpec(
          mode=mode,
          loss=cross_entropy,
          train_op=train_op,
          training_hooks=train_hook_list)

# MNIST Classifier Estimator
def MNIST_classifier_estimator(_):

    # Load training and eval data
    mnist = tf.contrib.learn.datasets.load_dataset("mnist")
    train_data = mnist.train.images  # Returns a np.array
    train_labels = np.asarray(mnist.train.labels, dtype=np.int32)
    eval_data = mnist.test.images  # Returns a np.array
    eval_labels = np.asarray(mnist.test.labels, dtype=np.int32)

    # Create a input function to train
    train_input_fn = tf.estimator.inputs.numpy_input_fn(
        x=train_data,
        y=train_labels,
        batch_size=_BATCH_SIZE,
        num_epochs=1,
        shuffle=True)
# Create a input function to eval
    eval_input_fn = tf.estimator.inputs.numpy_input_fn(
        x=eval_data,
        y=eval_labels,
        batch_size=_BATCH_SIZE,
        num_epochs=1,
        shuffle=False)
# Create a estimator with model_fn
    image_classifier = tf.estimator.Estimator(model_fn=model_fn,
                       model_dir=_MODEL_DIR)
# Finally, train and evaluate the model after each epoch
    for _ in range(_NUM_EPOCHS):
        image_classifier.train(input_fn=train_input_fn)
        metrics = image_classifier.evaluate(input_fn=eval_input_fn)

if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
#Run the code
tf.compat.v1.app.run(MNIST_classifier_estimator)
