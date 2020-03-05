from imports import *
import layers

def Resnet9(c=64, n_classes=10) -> Callable:
    model = layers.Sequential()
    model.add(layers.ConvBN(c))
    model.add(layers.ConvResBlk(c*2, res_convs=2))
    model.add(layers.ConvResBlk(c*4, res_convs=2))
    model.add(layers.ConvBlk(c*4))
    model.add(tf.keras.layers.GlobalMaxPool2D())
    model.add(layers.Classifier(n_classes))
    return model
