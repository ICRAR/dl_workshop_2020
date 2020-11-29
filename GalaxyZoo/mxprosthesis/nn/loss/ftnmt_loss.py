"""
Fractal Tanimoto (with dual) loss 
"""

from mxnet.gluon.loss import Loss                                                                                                                          

from mxnet import np as FF
from mxnet import npx as FFx

class ftnmt_loss(Loss):
    """
    This function calculates the average fractal tanimoto similarity for d = 0...depth
    """                                                                                          
    def __init__(self, depth=5, axis= [1,2,3], smooth = 1.0e-5, batch_axis=0, weight=None, **kwargs):
        super().__init__(batch_axis, weight, **kwargs)
        
        assert depth>= 0, ValueError("depth must be >= 0, aborting...")
        
        self.smooth = smooth
        self.axis=axis
        self.depth = depth

        if depth == 0:
            self.depth = 1
            self.scale = 1.
        else:
            self.depth = depth
            self.scale = 1./depth

    def inner_prod(self, prob, label):
        prod = prob*label
        prod = FF.sum(prod,axis=self.axis)

        return prod

    def tnmt_base(self, preds, labels):

        tpl  = self.inner_prod(preds,labels)
        tpp  = self.inner_prod(preds,preds)
        tll  = self.inner_prod(labels,labels)
        
       
        num = tpl + self.smooth
        scale = 1./self.depth
        denum = 0.0
        for d in range(self.depth):
            a = 2.**d
            b = -(2.*a-1.)

            denum = denum + FF.reciprocal( a*(tpp+tll) + b *tpl + self.smooth)

        result =  num * denum * scale       
        return  result 
                                                                                                                           
    def forward(self, preds, labels):

        l1 = self.tnmt_base(preds,labels)
        l2 = self.tnmt_base(1.-preds, 1.-labels)
         
        result = 0.5*(l1+l2) 
         
        return  1. - result
    
