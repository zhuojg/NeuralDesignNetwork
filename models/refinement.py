import tensorflow as tf
from tensorflow import keras
from models.graph import GraphTripleConvStack
from models.layers import build_mlp
import tensorflow_probability as tfp


class NDNRefinement(keras.Model):
    def __init__(self):
        super(NDNRefinement, self).__init__()
    
    def call(self):
        pass
