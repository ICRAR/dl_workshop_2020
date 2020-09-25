from mxnet import gluon
from mxnet.gluon import HybridBlock
from mxprosthesis.nn.layers.conv2Dnormed import *
from mxprosthesis.utils.get_norm import *
from mxprosthesis.nn.layers.attention import *

from mxnet import np as FF
from mxnet import npx as FFx

class ResNet_v2_block(HybridBlock):
    """
    ResNet v2 building block. It is built upon the assumption of ODD kernel 
    """
    def __init__(self, _nfilters,_kernel_size=(3,3),_dilation_rate=(1,1), 
                 _norm_type='BatchNorm', norm_groups=None, ngroups=1, **kwards):
        super().__init__(**kwards)

        self.nfilters = _nfilters
        self.kernel_size = _kernel_size
        self.dilation_rate = _dilation_rate

        # Ensures padding = 'SAME' for ODD kernel selection 
        p0 = self.dilation_rate[0] * (self.kernel_size[0] - 1)/2 
        p1 = self.dilation_rate[1] * (self.kernel_size[1] - 1)/2 
        p = (int(p0),int(p1))


        self.BN1 = get_norm(_norm_type, norm_groups=norm_groups )
        self.conv1 = gluon.nn.Conv2D(self.nfilters,kernel_size = self.kernel_size,padding=p,dilation=self.dilation_rate,use_bias=False,groups=ngroups)
        self.BN2 = get_norm(_norm_type, norm_groups= norm_groups)
        self.conv2 = gluon.nn.Conv2D(self.nfilters,kernel_size = self.kernel_size,padding=p,dilation=self.dilation_rate,use_bias=True, groups=ngroups)


    def forward(self,_input_layer):
 
        x = self.BN1(_input_layer)
        x = FFx.relu(x)
        x = self.conv1(x)

        x = self.BN2(x)
        x = FFx.relu(x)
        x = self.conv2(x)

        return x 


class FracTALResNet_unit(HybridBlock):
    def __init__(self, nfilters, ngroups=1, nheads=1, kernel_size=(3,3), dilation_rate=(1,1), norm_type = 'BatchNorm', norm_groups=None, ftdepth=5,**kwards):
        super().__init__(**kwards)

        self.block1 = ResNet_v2_block(nfilters,kernel_size,dilation_rate,_norm_type = norm_type, norm_groups=norm_groups, ngroups=ngroups)
        self.attn = FTAttention2D(nkeys=nfilters, nheads=nheads, kernel_size=kernel_size, norm = norm_type, norm_groups = norm_groups,ftdepth=ftdepth)

        self.gamma  = gluon.Parameter('gamma', shape=(1,), init=mx.init.Zero())

    def forward(self, input):
        out1 = self.block1(input)

        att2 = self.attn(input)
        att2 = self.gamma.data() * att2
        
        out  = (input + out1) * (FF.ones_like(out1) + att2) 
        return out 


class FracTALResNet_unit_v2(HybridBlock):
    def __init__(self, nfilters, ngroups=1, nheads=1, kernel_size=(3,3), dilation_rate=(1,1), norm_type = 'BatchNorm', norm_groups=None, ftdepth=5,**kwards):
        super().__init__(**kwards)

        self.block1 = ResNet_v2_block(nfilters,kernel_size,dilation_rate,_norm_type = norm_type, norm_groups=norm_groups, ngroups=ngroups)
        self.attn1 = FTAttention2D(nkeys=nfilters, nheads=nheads, kernel_size=kernel_size, norm = norm_type, norm_groups = norm_groups,ftdepth=ftdepth)
        self.attn2 = FTAttention2D(nkeys=nfilters, nheads=nheads, kernel_size=kernel_size, norm = norm_type, norm_groups = norm_groups,ftdepth=ftdepth)

        self.gamma1  = gluon.Parameter('gamma1', shape=(1,), init=mx.init.Zero())
        self.gamma2  = gluon.Parameter('gamma2', shape=(1,), init=mx.init.Zero())

    def forward(self, input):

        out1 = self.block1(input)
        att1 = self.attn1(out1)
        att1 = self.gamma1.data() * att1
        out1 = out1 * (FF.ones_like(out1) + att1) 

        att2 = self.attn2(input)
        att2 = self.gamma2.data() * att2 
        
        out  = (input + out1) * (FF.ones_like(out1) + att2) 
        return out 









