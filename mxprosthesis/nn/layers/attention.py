from mxnet import gluon
from mxnet.gluon import HybridBlock
from mxprosthesis.nn.layers.conv2Dnormed import *
from mxprosthesis.nn.layers.ftnmt import * 

from mxnet import np as FF
from mxnet import npx as FFx
       
class RelFTAttention2D(HybridBlock):
    def __init__(self, nkeys, kernel_size=3, padding=1,nheads=1, norm = 'BatchNorm', norm_groups=None,ftdepth=5,**kwards):
        super().__init__(**kwards)


        self.query  = Conv2DNormed(channels=nkeys,kernel_size= kernel_size, padding = padding, _norm_type= norm, norm_groups=norm_groups)
        self.key    = Conv2DNormed(channels=nkeys,kernel_size= kernel_size, padding = padding, _norm_type= norm, norm_groups=norm_groups)
        self.value  = Conv2DNormed(channels=nkeys,kernel_size= kernel_size, padding = padding, _norm_type= norm, norm_groups=norm_groups)

            
        self.metric_channel = FTanimoto(depth=ftdepth, axis=[2,3])
        self.metric_space = FTanimoto(depth=ftdepth, axis=1)
             
        self.norm = get_norm(name=norm, axis=1, norm_groups= norm_groups)
            
    def forward(self,  input1, input2, input3):

        # These should work with ReLU as well 
        q = FFx.sigmoid(self.query(input1))
        k = FFx.sigmoid(self.key(input2))# B,C,H,W 
        v = FFx.sigmoid(self.value(input3)) # B,C,H,W

        att_spat =  self.metric_space(q,k) # B,1,H,W 
        v_spat   =  att_spat * v # emphasize spatial features

        att_chan =  self.metric_channel(q,k) # B,C,1,1
        v_chan   =  att_chan * v # emphasize spatial features


        v_cspat =   0.5*(v_chan + v_spat) # emphasize spatial features
        v_cspat = self.norm(v_cspat)

        return v_cspat



class FTAttention2D(HybridBlock):
    def __init__(self, nkeys, kernel_size=3, padding=1, nheads=1, norm = 'BatchNorm', norm_groups=None,ftdepth=5,**kwards):
        super().__init__(**kwards)
        
        self. att = RelFTAttention2D(nkeys=nkeys,kernel_size=kernel_size, padding=padding, nheads=nheads, norm = norm, norm_groups=norm_groups,ftdepth=ftdepth,**kwards)


    def forward(self, input):
        return self.att(input,input,input)










