from mxnet.gluon import HybridBlock

from mxnet import np as FF
from mxnet import npx as FFx

class FTanimoto(HybridBlock):
    """
    This is the average fractal Tanimoto set similarity with complement. 
    """
    def __init__(self, depth=5, smooth=1.0e-5, axis=[2,3],**kwards):
        super().__init__(**kwards)
        
        assert depth >= 0, "Expecting depth >= 0, aborting ..."

        if depth == 0:
            self.depth = 1
            self.scale = 1.
        else:
            self.depth = depth
            self.scale = 1./depth
        
        self.smooth = smooth
        self.axis=axis
        
    def inner_prod(self, prob, label):
        prod = prob * label
        prod = FF.sum(prod,axis=self.axis,keepdims=True)

        return prod

        

    def tnmt_base(self, preds, labels):

        tpl  = self.inner_prod(preds,labels)
        tpp  = self.inner_prod(preds,preds)
        tll  = self.inner_prod(labels,labels)
        
       
        num = tpl + self.smooth
        denum = 0.0


        for d in range(self.depth):
            a = 2.**d
            b = -(2.*a-1.)

            denum = denum + FF.reciprocal(a*(tpp+tll) + b *tpl + self.smooth)

        return  num * denum * self.scale

    def forward(self, preds, labels):
            l12 = self.tnmt_base(preds,labels)
            l12 = l12 + self.tnmt_base(1.-preds, 1.-labels)

            return 0.5*l12
