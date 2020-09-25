from mxnet import gluon
from mxnet.gluon import HybridBlock

from mxprosthesis.nn.layers.scale import *
from mxprosthesis.nn.layers.conv2Dnormed import *

from mxnet import np as FF
from mxnet import npx as FFx

"""
For combining layers with Fusion (i.e. relative attention), see ../units/mxprosthesis.py
"""


class combine_layers(HybridBlock):
    def __init__(self,_nfilters,  _norm_type = 'BatchNorm', norm_groups=None, **kwards):
        HybridBlock.__init__(self,**kwards)
        

        # This performs convolution, no BatchNormalization. No need for bias. 
        self.up = UpSample(_nfilters, _norm_type = _norm_type, norm_groups=norm_groups) 

        self.conv_normed = Conv2DNormed(channels = _nfilters, 
                                            kernel_size=(1,1),
                                            padding=(0,0), 
                                            _norm_type=_norm_type,
                                            norm_groups=norm_groups)

        
            
        
    def forward(self,_layer_lo, _layer_hi):
        
        up = self.up(_layer_lo)
        up = FFx.relu(up)
        x = FF.concatenate([up,_layer_hi], axis=1)
        x = self.conv_normed(x)
        
        return x



