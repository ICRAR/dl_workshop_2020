from mxnet import gluon
from mxnet.gluon import HybridBlock

from mxprosthesis.nn.layers.conv2Dnormed import *
from mxprosthesis.nn.layers.attention import * 
from mxprosthesis.nn.pooling.psp_pooling import *


from mxprosthesis.nn.layers.scale import *
from mxprosthesis.nn.layers.combine import *

# CEEC units 
from mxprosthesis.nn.units.ceecnet import *

# FracTALResUnit
from mxprosthesis.nn.units.fractal_resnet import *  

from mxnet import np as FF
from mxnet import npx as FFx


# Purely encoder classifier, it requires an additional classification HEAD to make it useful.
class X_dn_encoder(HybridBlock):
    def __init__(self, nfilters_init, depth, widths=[3], psp_depth=4, verbose=True, norm_type='BatchNorm', norm_groups=None, nheads_start=8,  model='CEECNetV1', ftdepth=5, **kwards):
        super().__init__(**kwards)
        

        self.depth = depth
        
        print("Building block model::{}".format(model))
        if len(widths) == 1 and depth != 1:
            widths = widths * depth
        else:
            assert depth == len(widths), ValueError("depth and length of widths must match, aborting ...")

        self.conv_first = Conv2DNormed(nfilters_init,kernel_size=(1,1), _norm_type = norm_type, norm_groups=norm_groups)
            
        # List of convolutions and pooling operators 
        self.convs_dn = gluon.nn.HybridSequential()
        self.pools = gluon.nn.HybridSequential()


        for idx in range(depth):
            nheads = nheads_start * 2**idx #
            nfilters = nfilters_init * 2 **idx
            if verbose:
                print ("depth:= {0}, nfilters: {1}, nheads::{2}, widths::{3}".format(idx,nfilters,nheads,widths[idx]))
            tnet = gluon.nn.HybridSequential()
            for _ in range(widths[idx]):
                if model == 'CEECNetV1':
                    tnet.add(CEEC_unit_v1(nfilters=nfilters, nheads = nheads, ngroups = nheads , norm_type = norm_type, norm_groups=norm_groups,ftdepth=ftdepth))
                elif model == 'CEECNetV2':
                    tnet.add(CEEC_unit_v2(nfilters=nfilters, nheads = nheads, ngroups = nheads , norm_type = norm_type, norm_groups=norm_groups,ftdepth=ftdepth))
                elif model == 'FracTALResNet':
                    tnet.add(FracTALResNet_unit(nfilters=nfilters, nheads = nheads, ngroups = nheads , norm_type = norm_type, norm_groups=norm_groups,ftdepth=ftdepth))
                elif model == 'FracTALResNetV2':
                    tnet.add(FracTALResNet_unit_v2(nfilters=nfilters, nheads = nheads, ngroups = nheads , norm_type = norm_type, norm_groups=norm_groups,ftdepth=ftdepth))
                else:
                    raise ValueError("I don't know requested model, available options: CEECNetV1, CEECNetV2, FracTALResNet - Given model::{}, aborting ...".format(model))
            self.convs_dn.add(tnet)

            if idx < depth-1:
                self.pools.add(DownSample(nfilters, _norm_type=norm_type, norm_groups=norm_groups)) 
        # Middle pooling operator 
        self.middle = PSP_Pooling(nfilters,depth=psp_depth, _norm_type=norm_type,norm_groups=norm_groups)
                               
   
    def forward(self, input):

        conv1_first = self.conv_first(input)
 
        # ******** Going down ***************
        pools = conv1_first
        for idx in range(self.depth):
            conv1 = self.convs_dn[idx](pools)
            if idx < self.depth-1:
                # Evaluate pools 
                pools =  self.pools[idx](conv1)
        # Middle psppooling
        middle =  self.middle(conv1)
        # Activation of middle layer
        middle = FFx.relu(middle)
        #print (middle.shape)      
        return middle
    

