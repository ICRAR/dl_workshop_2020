import mxnet as mx
from mxnet import gluon, nd, autograd
from gluoncv import utils
import time, os, math, argparse


parser = argparse.ArgumentParser(description='MXNet Gluon GZoo training')
parser.add_argument('--root', type=str, default=r'/Users/foivos/Documents/kaggle_comps/',
                    help='root directory that contains GalaxyZoo code and data')
parser.add_argument('--batch-size', type=int, default=128,
                    help='batch size for training and testing (default: 128)')
parser.add_argument('--epochs', type=int, default=600,
                    help='number of epochs to train (default: 600)')
parser.add_argument('--lr', type=float, default=0.001,
                    help='learning rate (default: 0.001)')
parser.add_argument('--cuda', action='store_true', default=True,
                    help='Train on GPU with CUDA')
parser.add_argument('--model',type=str, default='resnet18_v2',
                    help='Model base for feature extraction, default::resnet18_v2')
parser.add_argument('--causal',type=bool, default=True,
                    help='Model predictions HEAD, causal vs standard, default:: causal=True')


opt = parser.parse_args()


import sys
sys.path.append(opt.root)

# Data augmentation definitions 
from mxnet.gluon.data.vision import transforms

# Model definitions 
from GalaxyZoo.mxnet.models.resnets import *
from GalaxyZoo.mxnet.models.xxnets import * 

if opt.causal:
    tname = r'causal'
flname_write = r'Results/'+ opt.model + r'_' + tname +r'.txt' 

# Decide on cuda: 
if opt.cuda and mx.util.get_gpu_count():
    ctx = mx.gpu()
else:
    ctx = mx.cpu()

# Define model
net = ResNet_v2(model_name = opt.model, causal = opt.causal,ctx=ctx)
net.hybridize()  # ZoomZoom!! 


# Data augmentation definitions 
transform_train = transforms.Compose([
    # Randomly crop an area, and then resize it to be 32x32
    transforms.RandomResizedCrop(400,scale=(0.6,1.)),
    # Randomly flip the image horizontally/vertically
    transforms.RandomFlipLeftRight(),
    transforms.RandomFlipTopBottom(),
    # Randomly jitter the brightness, contrast and saturation of the image
    transforms.RandomColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
    # Transpose the image from height*width*num_channels to num_channels*height*width
    # and map values from [0, 255] to [0,1]
    transforms.ToTensor(),
    # Normalize the image with mean and standard deviation calculated across all images
    transforms.Normalize([11.663384, 10.260227,  7.65015 ], [21.421959, 18.044296, 15.494861])
])

transform_test = transforms.Compose([
    transforms.Resize(400),
    transforms.ToTensor(),
    transforms.Normalize([11.663384, 10.260227,  7.65015 ], [21.421959, 18.044296, 15.494861])
])

# Datasets/DataLoaders 
from GalaxyZoo.mxnet.src.GZooDataset import *
dataset_train = GZooData(root=os.path.join(opt.root,'GalaxyZoo'), transform=transform_train)
datagen_train = gluon.data.DataLoader(dataset_train,batch_size=opt.batch_size,shuffle=True)

dataset_dev = GZooData(root=os.path.join(opt.root,'GalaxyZoo'), mode='dev')
datagen_dev = gluon.data.DataLoader(dataset_dev,batch_size=opt.batch_size,shuffle=False,last_batch='drop')


# Adam parameters                                                                                  
optimizer = 'Adam'                                                                                 
lr = opt.lr
# *********************************************************************************************    
# Epochs in which we want to step                                                                  
steps_epochs = [200,350]                                                                           
# assuming we keep partial batches, see `last_batch` parameter of DataLoader                       
iterations_per_epoch = math.ceil(len(dataset_train) / opt.batch_size)                                  
# iterations just before starts of epochs (iterations are 1-indexed)                               
steps_iterations = [s*iterations_per_epoch for s in steps_epochs]                                  
scheduler = mx.lr_scheduler.MultiFactorScheduler(base_lr=lr, step= steps_iterations, factor=0.1)   
# **********************************************************************************************   
optimizer_params = {'learning_rate': lr,'lr_scheduler':scheduler}                                  
trainer = gluon.Trainer(net.collect_params(), optimizer, optimizer_params)                         



train_metric = mx.metric.MSE()
loss_fn  = gluon.loss.L1Loss() 

# development metric: 
def test(tctx, tnet, tdatagen_dev):
    metric = mx.metric.Accuracy()
    for data in tdatagen_dev:
        #data = gluon.utils.split_and_load(batch[0], ctx_list=ctx, batch_axis=0)
        imgs, labels = data
        imgs = imgs.as_in_context(tctx)
        # outputs = [net(X) for X in data]
        #outputs = nd.concatenate(outputs,axis=0)
        with mx.autograd.predict_mode():
            preds = net(imgs)
        metric.update(preds=preds, labels=labels)
    return metric.get()



def train(epochs,ctx,flname_write):

    with open(flname_write,"w") as f:
        print('epoch','train_mse','val_mse','train_loss',file=f,flush=True)


        for epoch in range(epochs):
            tic = time.time()
            train_metric.reset()
            train_loss = 0

            # Loop through each batch of training data
            for i, (data,label) in enumerate(datagen_train):
                # Extract data and label
                data = data.as_in_context(ctx)
                label = label.as_in_context(ctx)


                # AutoGrad
                with autograd.record():
                    output = net(data)
                    loss = loss_fn(output, label).mean()
            
                # Backpropagation
                loss.backward()

                # Optimize
                trainer.step(opt.batch_size) # This is the batch_size

                # Update metrics
                train_loss += loss.asscalar()
                train_metric.update(label, output)
            train_loss = train_loss / len(datagen_train)
            name, train_mse = train_metric.get()
            # Evaluate on Validation data
            nd.waitall() # necessary to avoid cuda malloc
            name, val_mse = test(ctx, net, datagen_dev)

            # Print metrics
            # print both on screen and in file 
            print('epoch={} train_mse={} val_mse={} train_loss={} time={}'.format(epoch, train_mse, val_mse, train_loss, time.time()-tic))
            print(epoch, train_mse, val_mse, train_loss, file=f,flush=True)



if __name__=='__main__':
    #print ("Passed first test")
    train(opt.epochs, ctx, flname_write)
