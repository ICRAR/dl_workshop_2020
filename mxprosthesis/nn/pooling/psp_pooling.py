from mxnet import gluon
from mxnet.gluon import  HybridBlock
from mxprosthesis.nn.layers.conv2Dnormed import *

from mxnet import np as FF
from mxnet import npx as FFx


class PSP_Pooling(gluon.HybridBlock):
    def __init__(self, nfilters, depth=4, _norm_type = 'BatchNorm', norm_groups=None, mob=False, **kwards):
        gluon.HybridBlock.__init__(self,**kwards)
        
               
        self.depth = depth 
        self.convs = gluon.nn.HybridSequential()
        for _ in range(depth):
            self.convs.add(Conv2DNormed(nfilters//self.depth,kernel_size=(1,1),padding=(0,0),_norm_type=_norm_type, norm_groups=norm_groups)) 
            
        self.conv_norm_final = Conv2DNormed(channels = nfilters,
                                            kernel_size=(1,1),
                                            padding=(0,0),
                                            _norm_type=_norm_type, 
                                            norm_groups=norm_groups)



    def forward(self,input):
        _, _, h, w = input.shape

        p = [input]
        for i in range(self.depth):
            hnew = h // (2**i)
            wnew = w // (2**i)
            kernel = (hnew,wnew)
            x = FFx.pooling(input,kernel=kernel, stride=kernel, pool_type='max')
            #x = mx.nd.UpSampling(x.as_nd_ndarray(),sample_type='nearest',scale=hnew) 
            x = mx.contrib.ndarray.BilinearResize2D(x.as_nd_ndarray(),height=h,width=w) 
            x = x.as_np_ndarray()
            x = self.convs[i](x)
            p += [x]

        out = FF.concatenate(p,axis=1)       
        out = self.conv_norm_final(out)


        return out



