from imports import *

# Sequential transforming function returning values for each layer
def sequential_transforms(x: tf.Tensor, tfms: List) -> tf.Tensor:
    update_func = lambda x, y: y(x)
    return functools.reduce(update_func, tfms, x)

# Kernel Initialization function borrowed from pytorch
def init_pytorch(shape, dtype=tf.float32, partition_info=None) -> tf.Tensor:
    fan = np.prod(shape[:-1])
    bound = 1 / math.sqrt(fan)
    return tf.random.uniform(shape, minval=-bound, maxval=bound, dtype=dtype)

# Sequential class for layers
class Sequential(tf.keras.Model):

    def __init__(self):
        super().__init__()
        self.f_layers = []

    def add(self, layer: tf.keras.layers.Layer):
        self.f_layers.append(layer)

    def __call__(self, x: tf.Tensor, *args, **kw_args) -> tf.Tensor:
        return sequential_transforms(x, self.f_layers)

# Conv-> BN-> ReLU Layer
class ConvBN(Sequential):
    def __init__(self, c: int, kernel_size=3, strides=(1, 1), kernel_initializer=init_pytorch,
                bn_mom=0.9,bn_eps=1e-05):
        super().__init__()
        self.add(tf.keras.layers.Conv2D(filters=c, kernel_size=kernel_size, strides=strides,
                                        kernel_initializer=kernel_initializer,
                                        padding='same', use_bias=False))
        self.add(tf.keras.layers.BatchNormalization(momentum=bn_mom, epsilon=bn_eps))
        self.add(tf.keras.layers.Activation('relu'))

# Conv-> BN-> ReLU -> Maxpool Layer
class ConvBlk(Sequential):
    def __init__(self, c, convs=1, kernel_size=3, kernel_initializer=init_pytorch,
                 bn_mom=0.9,bn_eps=1e-05):
        super().__init__()
        for i in range(convs):
            self.add(
                ConvBN(c, kernel_size=kernel_size, kernel_initializer=kernel_initializer,
                        bn_mom=bn_mom, bn_eps=bn_eps))
        self.add(tf.keras.layers.MaxPooling2D(2))

# Residual Layers
class ConvResBlk(ConvBlk):
    def __init__(self, c, convs=1, res_convs=2, kernel_size=3, kernel_initializer=init_pytorch,
                 bn_mom=0.9, bn_eps=1e-05):
        super().__init__(c, convs=convs, kernel_size=kernel_size,
                         bn_mom=bn_mom, bn_eps=bn_eps)
        self.res = []
        for i in range(res_convs):
            conv_bn = ConvBN(c, kernel_size=kernel_size, kernel_initializer=kernel_initializer,
                            bn_mom=bn_mom,bn_eps=bn_eps)
            self.res.append(conv_bn)

    def __call__(self, x: tf.Tensor, *args, **kw_args) -> tf.Tensor:
        h = super().__call__(x)
        hh = sequential_transforms(h, self.res)
        return h + hh

# Scaling function for scaling last classifier layer
class Scaling(tf.keras.layers.Layer):
    def __init__(self, weight: float):
        super().__init__()
        self.weight = weight

    def __call__(self, x: tf.Tensor, *args, **kw_args) -> tf.Tensor:
        return x * self.weight

# Final Classifer layer with a scaling factor of 0.2
class Classifier(Sequential):
    def __init__(self, n_classes: int, kernel_initializer = init_pytorch, weight = 0.2):
        super().__init__()
        self.add(tf.keras.layers.Dense(n_classes, kernel_initializer=kernel_initializer, use_bias=False))
        self.add(Scaling(weight))
