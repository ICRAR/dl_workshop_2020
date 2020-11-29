import mxnet as mx
from mxnet import gluon
from mxnet.gluon import nn, HybridBlock

from GalaxyZoo.models.heads.GZooHEAD   import *

from mxprosthesis.models.classification.x_dn_features import *

class xxnets(HybridBlock):
    def __init__(self, nfilters_init=64, depth=3, widths=[2], nheads_start =128//8, in_features=2048, NClasses=37,causal=True, psp_depth=4, model_name = 'CEECNetV1', causality_model='v3',ftdepth=0,  **kwards):
        super().__init__(**kwards)
        self.features = X_dn_encoder(model=model_name, nfilters_init=nfilters_init,depth=depth,widths=widths,psp_depth=psp_depth,nheads_start=nheads_start)
        self.causality_model = causality_model 

        if causal:
            if causality_model =='v1':
                self.head = GZooHEAD(in_features=in_features)
        else:
           self.head = nn.Dense(units=NClasses,activation='sigmoid')
        # Necessary to compress features for memory convservation - reduce img size, keep only
        # channels 
        self.compress = gluon.nn.HybridSequential()
        self.compress.add(gluon.nn.Dense(units=1,flatten=False,use_bias=False))
        self.compress.add(gluon.nn.BatchNorm())

    def forward(self, input):
        out = self.features(input)
        shape = out.shape
        out = out.reshape(*(shape[:2]),-1)
        out = self.compress(out)
        out = mx.npx.relu(out)
        out = out.squeeze()
        
        # Here conditioning takes place
        out = self.head(out)
        return out



