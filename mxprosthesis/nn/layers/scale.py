from mxnet  import gluon
from mxnet.gluon import HybridBlock

from mxprosthesis.nn.layers.conv2Dnormed import * 
from mxprosthesis.utils.get_norm import * 

import mxnet as mx 
from mxnet import np as FF
from mxnet import npx as FFx

class DownSample(HybridBlock):
    def __init__(self, nfilters, factor=2,  _norm_type='BatchNorm', norm_groups=None, **kwargs): 
        super().__init__(**kwargs)
        
        
        # Double the size of filters, since you downscale by 2. 
        self.factor = factor 
        self.nfilters = nfilters * self.factor

        self.kernel_size = (3,3) 
        self.strides = (factor,factor)
        self.pad = (1,1)

        self.convdn = Conv2DNormed(self.nfilters,
                    kernel_size=self.kernel_size,
                    strides=self.strides,
                    padding=self.pad,
                    _norm_type = _norm_type, 
                    norm_groups=norm_groups)
 
    
    def forward(self,input):
        
        x = self.convdn(input)

        return x 


class UpSample(HybridBlock):
    def __init__(self,nfilters, factor = 2,  _norm_type='BatchNorm', norm_groups=None, **kwards):
        HybridBlock.__init__(self,**kwards)
        
        
        self.factor = factor
        self.nfilters = nfilters // self.factor
        
        self.convup_normed = Conv2DNormed(self.nfilters,
                                              kernel_size = (1,1),
                                              _norm_type = _norm_type, 
                                              norm_groups=norm_groups)
    
    def forward(self,_xl):
        x = mx.nd.UpSampling(_xl.as_nd_ndarray(), scale=self.factor, sample_type='nearest')
        x = x.as_np_ndarray()
        x = self.convup_normed(x)
        
        return x

