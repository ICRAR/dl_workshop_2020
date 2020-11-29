import mxnet as mx
from mxnet import gluon
from mxnet.gluon import HybridBlock

from mxnet import np as FF
from mxnet import npx as FFx

class SigmoidCrisp(HybridBlock):
    def __init__(self, smooth=1.e-2,**kwards):
        super().__init__(**kwards)


        self.smooth = smooth
        self.gamma  = gluon.Parameter('gamma', shape=(1,), init=mx.init.One())


    def forward(self, input):
            out = self.smooth + FFx.sigmoid(self.gamma.data())
            out = FF.reciprocal(out)

            out = input*out
            out = FFx.sigmoid(out)
            return out 



