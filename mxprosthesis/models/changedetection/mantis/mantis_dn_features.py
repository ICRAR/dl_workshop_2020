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

"""
if upFuse == True, then instead of concatenation of the encoder features with the decoder features, the algorithm performs Fusion with 
relative attention.  
"""


class mantis_dn_features(HybridBlock):
    def __init__(self, nfilters_init, depth, widths=[1], psp_depth=4, verbose=True, norm_type='BatchNorm', norm_groups=None, nheads_start=8,  model='CEECNetV1', upFuse=False, ftdepth=5, **kwards):
        super().__init__(**kwards)
        

        self.depth = depth


        if len(widths) == 1 and depth != 1:
            widths = widths * depth
        else:
            assert depth == len(widths), ValueError("depth and length of widths must match, aborting ...")

        self.conv_first = Conv2DNormed(nfilters_init,kernel_size=(1,1), _norm_type = norm_type, norm_groups=norm_groups)
        self.fuse_first = Fusion(nfilters_init, norm=norm_type, norm_groups=norm_groups)
            
        # List of convolutions and pooling operators 
        self.convs_dn = gluon.nn.HybridSequential()
        self.pools = gluon.nn.HybridSequential()
        self.fuse  = gluon.nn.HybridSequential()


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
                else:
                    raise ValueError("I don't know requested model, aborting ... - Given model::{}".format(model))
            self.convs_dn.add(tnet)

            if idx < depth-1:
                self.fuse.add( Fusion( nfilters=nfilters, nheads = nheads , norm = norm_type, norm_groups=norm_groups)  )
                self.pools.add(DownSample(nfilters, _norm_type=norm_type, norm_groups=norm_groups)) 
        # Middle pooling operator 
        self.middle = PSP_Pooling(nfilters,depth=psp_depth, _norm_type=norm_type,norm_groups=norm_groups)
                               
            
        self.convs_up = gluon.nn.HybridSequential() # 1 argument
        self.UpCombs = gluon.nn.HybridSequential() # 2 arguments
        for idx in range(depth-1,0,-1):
            nheads = nheads_start * 2**idx 
            nfilters = nfilters_init * 2 **(idx-1)
            if verbose:
               print ("depth:= {0}, nfilters: {1}, nheads::{2}, widths::{3}".format(2*depth-idx-1,nfilters,nheads,widths[idx]))
                
            tnet = gluon.nn.HybridSequential()
            for _ in range(widths[idx]):
                if model == 'CEECNetV1':
                    tnet.add(CEEC_unit_v1(nfilters=nfilters, nheads = nheads, ngroups = nheads , norm_type = norm_type, norm_groups=norm_groups,ftdepth=ftdepth))
                elif model == 'CEECNetV2':
                    tnet.add(CEEC_unit_v2(nfilters=nfilters, nheads = nheads, ngroups = nheads , norm_type = norm_type, norm_groups=norm_groups,ftdepth=ftdepth))
                elif model == 'FracTALResNet':
                    tnet.add(FracTALResNet_unit(nfilters=nfilters, nheads = nheads, ngroups = nheads , norm_type = norm_type, norm_groups=norm_groups,ftdepth=ftdepth))
                else:
                    raise ValueError("I don't know requested model, aborting ... - Given model::{}".format(model))
            self.convs_up.add(tnet)
                
            if upFuse==True:
                self.UpCombs.add(combine_layers_wthFusion(nfilters=nfilters, nheads=nheads, _norm_type=norm_type,norm_groups=norm_groups,ftdepth=ftdepth))
            else:
                self.UpCombs.add(combine_layers(nfilters, _norm_type=norm_type,norm_groups=norm_groups))
                
    def forward(self, input_t1, input_t2):

        conv1_t1 = self.conv_first(input_t1)
        conv1_t2 = self.conv_first(input_t2)
            
        fuse1 = self.fuse_first(conv1_t1,conv1_t2)
        
        # ******** Going down ***************
        fusions   = []

        pools1 = conv1_t1
        pools2 = conv1_t2 

        for idx in range(self.depth):
            conv1 = self.convs_dn[idx](pools1)
            conv2 = self.convs_dn[idx](pools2)

            if idx < self.depth-1:
                # Evaluate fusions 
                conv1 = conv1
                conv2 = conv2
                fusions = fusions + [self.fuse[idx](conv1,conv2)]
                # Evaluate pools 
                pools1 =  self.pools[idx](conv1)
                pools2 =  self.pools[idx](conv2)

        # Middle psppooling
        middle =  self.middle(FF.concatenate([conv1,conv2], axis=1))
        # Activation of middle layer
        middle = FFx.relu(middle)
        fusions   = fusions + [middle] 

        # ******* Coming up ****************
        convs_up = middle
        for idx in range(self.depth-1):
            convs_up = self.UpCombs[idx](convs_up, fusions[-idx-2])
            convs_up = self.convs_up[idx](convs_up)
            
        return convs_up, fuse1
    

