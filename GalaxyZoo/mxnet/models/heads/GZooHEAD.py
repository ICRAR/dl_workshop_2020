import mxnet as mx
from mxnet import gluon
from mxnet.gluon import nn, HybridBlock

from mxprosthesis.nn.layers.densenormed import *

class GZooHEADLayer(HybridBlock):
    def __init__(self, mid_features, out_features, **kwards):
        super().__init__(**kwards)

        self.d1 = DenseNormed(units=mid_features)
        self.d2 = gluon.nn.Dense(units=out_features)


    def forward(self,input):
        out = self.d1(input)
        out = mx.npx.relu(out)
        out = self.d2(out)

        return out


class GZooHEAD(HybridBlock):
    def __init__(self,in_features=512, **kwards):
        super().__init__(**kwards)




        self.q1  = GZooHEADLayer(mid_features=in_features, out_features=3) # softmax
        self.q1bal  = gluon.nn.Dense(units=in_features) # softmax

        self.q2  = GZooHEADLayer(mid_features=in_features, out_features=2) # sigmoid
        self.q2bal  = gluon.nn.Dense(units=in_features) # softmax

        self.q3  = GZooHEADLayer(mid_features=in_features, out_features=2) # sigmoid
        self.q3bal  = gluon.nn.Dense(units=in_features) # softmax

        self.q4  = GZooHEADLayer(mid_features=in_features, out_features=2) # sigmoid
        self.q4bal  = gluon.nn.Dense(units=in_features) # softmax

        self.q5  = GZooHEADLayer(mid_features=in_features, out_features=4) # sigmoid

        self.q6  = GZooHEADLayer(mid_features=in_features, out_features=2) # sigmoid
        self.q6bal  = gluon.nn.Dense(units=in_features) # softmax

        self.q7  = GZooHEADLayer(mid_features=in_features, out_features=3) # sigmoid
        self.q7bal  = gluon.nn.Dense(units=in_features) # softmax

        self.q8  = GZooHEADLayer(mid_features=in_features, out_features=7) # sigmoid

        self.q9  = GZooHEADLayer(mid_features=in_features, out_features=3) # sigmoid
        self.q9bal  = gluon.nn.Dense(units=in_features) # softmax

        self.q10 = GZooHEADLayer(mid_features=in_features, out_features=3) # sigmoid
        self.q10bal  = gluon.nn.Dense(units=in_features) # softmax

        self.q11 = GZooHEADLayer(mid_features=in_features, out_features=6) # sigmoid
        self.q11bal  = gluon.nn.Dense(units=in_features) # softmax

    def forward(self, input):


        q1 = self.q1(input)
        q1 = mx.npx.softmax(q1,axis=-1)
        q1bal = self.q1bal(q1)

        q2in = mx.np.concatenate([q1bal,input],axis=-1)
        q2 = self.q2 (q2in)
        q2 = mx.npx.sigmoid(q2)
        q2bal = self.q2bal(q2)
        q2bal = mx.npx.relu(q2bal)

        q3in = mx.np.concatenate([input,q2bal],axis=-1)
        q3 = self.q3(q3in)
        q3 = mx.npx.sigmoid(q3)
        q3bal = self.q3bal(q3)
        q3bal = mx.npx.relu(q3bal)

        q4in = mx.np.concatenate([input,q3bal],axis=-1)
        q4 = self.q4(q4in)
        q4 = mx.npx.sigmoid(q4)
        q4bal = self.q4bal(q4)
        q4bal = mx.npx.relu(q4bal)


        q10in = mx.np.concatenate([input,q4bal],axis=-1)
        q10 = self.q10(q10in)
        q10 = mx.npx.sigmoid(q10)
        q10bal = self.q10bal(q10)
        q10bal = mx.npx.relu(q10bal)

        q11in = mx.np.concatenate([input,q10bal],axis=-1)
        q11 = self.q11(q11in)
        q11 = mx.npx.sigmoid(q11)
        q11bal = self.q11bal(q11)
        q11bal = mx.npx.relu(q11bal)

        q5in = mx.np.concatenate([input,q4bal,q11bal],axis=-1)
        q5 = self.q5(q5in)
        q5 = mx.npx.sigmoid(q5)

        q7in = mx.np.concatenate([input,q1bal],axis=-1)
        q7 = self.q7(q7in)
        q7 = mx.npx.sigmoid(q7)
        q7bal = self.q7bal(q7)
        q7bal = mx.npx.relu(q7bal)

        q9in = mx.np.concatenate([input,q2bal],axis=-1)
        q9 = self.q9(q9in)
        q9 = mx.npx.sigmoid(q9)
        q9bal = self.q9bal(q9)
        q9bal = mx.npx.relu(q9bal)


        q6in = mx.np.concatenate([input,q7bal,q9bal],axis=-1)
        q6 = self.q6(q6in)
        q6 = mx.npx.sigmoid(q6)
        q6bal = self.q6bal(q6)
        q6bal = mx.npx.relu(q6bal)


        q8in = mx.np.concatenate([input,q6bal],axis=-1)
        q8 = self.q8(q8in)
        q8 = mx.npx.sigmoid(q8)


        qall = mx.np.concatenate([q1,q2,q3,q4,q5,q6,q7,q8,q9,q10,q11],axis=-1)
        return qall
