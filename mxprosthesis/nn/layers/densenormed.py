import mxnet as mx
from mxnet import gluon
from mxnet.gluon import HybridBlock

from mxnet import np as FF
from mxnet import npx as FFx

class DenseNormed(HybridBlock):
    def __init__(self, units,  flatten=True,dtype='float32', weight_initializer=None, in_units=0, **kwargs):
        super().__init__(**kwargs)


        self.dense = gluon.nn.Dense(units=units,  use_bias=False, flatten=flatten, dtype=dtype, weight_initializer=weight_initializer, in_units=in_units, **kwargs)
        # Add option for normalization variation
        self.norm = gluon.nn.BatchNorm()


    def forward(self,  input):
        out = self.dense(input)
        out = self.norm(out)
        return out 

