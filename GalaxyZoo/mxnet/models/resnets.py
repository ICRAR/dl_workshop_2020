import mxnet as mx
from mxnet import gluon
from mxnet.gluon import nn, HybridBlock
import gluoncv

from GalaxyZoo.mxnet.models.heads.GZooHEAD import *

class ResNet_v2(HybridBlock):
    def __init__(self, ctx, NClasses=37,pretrained=False,causal=True, model_name = 'resnet18_v2',  **kwards):
        super().__init__(**kwards)
        viable_names = {'resnet18_v2', 'resnet34_v2', 'resnet50_v2', 'resnet101_v2','resnet152_v2'}
        assert model_name in viable_names, ValueError("Was given model name, not compatible with this construct, aborting... Viable names::{}".format(viable_names))
        if pretrained == False:
            model = gluoncv.model_zoo.get_model(model_name,pretrained=pretrained,ctx=ctx)
            self.features = model.features
            if causal:
                self.output = GZooHEAD()
            else:
                self.output = nn.Dense(units=NClasses,activation='sigmoid')
        
            self.features.initialize(ctx=ctx)
            self.output.initialize(ctx=ctx)
        
        elif pretrained == True:
            model = gluoncv.model_zoo.get_model(model_name,pretrained=pretrained,ctx=ctx)
            self.features = model.features
            if causal:
                self.output = GZooHEAD()
                self.output.initialize(ctx=ctx)
            else:
                self.model.output = nn.Dense(units=NClasses,activation='sigmoid')
                self.output.initialize(ctx=ctx)


    def hybrid_forward(self, F, input):
        out = self.features(input)
        out = self.output(out)
        return out
