from imports import *

class SGD(tf.compat.v1.train.MomentumOptimizer):
    def __init__(self, lr: tf.Tensor, mom: float, wd: float):
        super().__init__(lr, momentum=mom, use_nesterov=True)
        self.wd = wd

    def compute_gradients(self, loss: tf.Tensor) -> List[Tuple[tf.Tensor, tf.Tensor]]:
        grads_and_vars = super().compute_gradients(loss)

        l = len(grads_and_vars)
        for i in range(l):
            g, v = grads_and_vars[i]
            g += v * self.wd
            grads_and_vars[i] = (g, v)

        return grads_and_vars
