import mxnet as mx
from mxnet import gluon
from mxnet.gluon import nn, HybridBlock
import gluoncv

from GalaxyZoo.mxnet.models.heads.GZooHEAD import *
from mxprosthesis.models.classification.x_dn_features import *

class xxnets(HybridBlock):
    def __init__(self, nfilters_init=64, depth=3, widths=[2], nheads_start =128//8, in_features=2048, NClasses=37,causal=True, psp_depth=3, model_name = 'CEECNetV1', **kwards):
        super().__init__(**kwards)
        self.features = X_dn_encoder(model=model_name, nfilters_init=nfilters_init,depth=depth,widths=widths,psp_depth=psp_depth,nheads_start=nheads_start)
        if causal:
           self.output = GZooHEAD(in_features=in_features)
        else:
           self.output = nn.Dense(units=NClasses,activation='sigmoid')
        
        self.flat = nn.Flatten()
    def hybrid_forward(self, F, input):
        out = self.features(input)
        out = self.flat(out)
        out = self.output(out)
        return out
