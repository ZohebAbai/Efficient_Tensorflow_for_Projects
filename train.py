from imports import *
from optimizers import *

# Linear decay function
def linear_decay() -> Callable:
    return functools.partial(tf.compat.v1.train.polynomial_decay, end_learning_rate=0.0, power=1.0, cycle=False)

# Initial warmup function
def warmup_lr_sched(step: tf.Tensor, warmup_steps: int, init_lr: float, lr) -> tf.Tensor:
    step = tf.cast(step, tf.float32)
    warmup_steps = tf.constant(warmup_steps, dtype=tf.float32)
    warmup_lr = init_lr * step / warmup_steps
    is_warmup = tf.cast(step < warmup_steps, tf.float32)
    return (1.0 - is_warmup) * lr + is_warmup * warmup_lr

# One cycle learning rate method with warmup for first warmup steps and then a linear decay to zero
def one_cycle_lr(init_lr: float, total_steps: int, warmup_steps: int, decay_sched: Callable) -> Callable:

    def lr_func(step: tf.Tensor = None) -> tf.Tensor:
        if step is None:
            step = tf.compat.v1.train.get_or_create_global_step()

        lr = tf.constant(value=init_lr, shape=[], dtype=tf.float32)
        lr = decay_sched(lr, step - warmup_steps, total_steps - warmup_steps)
        return lr if warmup_steps == 0 else warmup_lr_sched(step, warmup_steps, init_lr, lr)

    return lr_func

# SGD optimizer
def sgd_optimizer(lr_func: Callable, mom: float = 0.9, wd: float = 0.0) -> Callable:
    def opt_func():
        lr = lr_func()
        return SGD(lr, mom=mom, wd=wd)
    return opt_func

# Function which returns every required specs to passed to tf.estimator class
def get_model_func(model_arch: Callable, opt_func: Callable, work_dir : str,
                    reduction: str = tf.compat.v1.losses.Reduction.MEAN) -> Callable:

    def model_func(features, labels, mode):
        tf.keras.backend.set_learning_phase(1)
        model = model_arch()
        logits = model(features)
        y_pred = tf.math.argmax(logits, axis=-1)
        probabilities = tf.nn.softmax(logits)
        var = None
        with tf.name_scope('loss'):
            ce_loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits, reduction=reduction)
            tf.summary.scalar('loss', ce_loss)
        with tf.name_scope('accuracy'):
            accuracy = tf.compat.v1.metrics.accuracy(labels=labels, predictions=y_pred)
            tf.summary.scalar('accuracy', accuracy[1])

        # PREDICT MODE
        predictions = {
            "predicted": y_pred,
            "probabilities": probabilities
            }
        if mode == tf.estimator.ModeKeys.PREDICT:
            return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

        # EVAL MODE
        if mode == tf.estimator.ModeKeys.EVAL:
            return tf.estimator.EstimatorSpec(mode=mode, loss=ce_loss,
                eval_metric_ops={'accuracy': accuracy}, evaluation_hooks=None)

        # TRAIN MODE
        if mode == tf.estimator.ModeKeys.TRAIN:
            step = tf.compat.v1.train.get_or_create_global_step()
            opt = opt_func()
            grads_and_vars = opt.compute_gradients(ce_loss) # by default var_list=tf.compat.v1.trainable_variables()             
            with tf.control_dependencies(model.get_updates_for(features)):
                train_op = opt.apply_gradients(grads_and_vars, global_step=step)

        # Create a hook to print loss every 100 iter.
        train_hook_list= []
        train_tensors_log = tf.train.LoggingTensorHook({'accuracy': accuracy[1],'loss': ce_loss,'global_step': step},
                                                        every_n_iter=100)
        train_profiler = tf.train.ProfilerHook(save_steps=100, output_dir= os.path.join(work_dir,"tracing"))
        train_hook_list.append(train_tensors_log)
        train_hook_list.append(train_profiler)
        return tf.estimator.EstimatorSpec(mode=mode, loss=ce_loss,
                                        train_op=train_op,
                                        training_hooks=train_hook_list)
    return model_func
