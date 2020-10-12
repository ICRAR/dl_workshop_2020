import mxnet as mx
from mxnet import gluon, nd, autograd
#from gluoncv import utils
import time, os, math, argparse

mx.npx.set_np()

parser = argparse.ArgumentParser(description='MXNet Gluon GZoo training')
parser.add_argument('--root', type=str, default=r'/home/foivos/Projects/dl_workshop_2020/',
                    help='root directory that contains GalaxyZoo code and data')
parser.add_argument('--batch-size', type=int, default=6,
                    help='batch size for training and testing (default:32)')
parser.add_argument('--crop-size', type=int, default=256, # this is not the best solution, but ...
                    help='crop size of input image, for memory efficiency(default:256)')
parser.add_argument('--epochs', type=int, default=600,
                    help='number of epochs to train (default: 600)')
parser.add_argument('--lr', type=float, default=0.001,
                    help='learning rate (default: 0.001)')
parser.add_argument('--cuda', action='store_true', default=True,
                    help='Train on GPU with CUDA')
parser.add_argument('--nfilters_init',type=int, default=32,
                    help='XX nfilters_init, default::32')
parser.add_argument('--model',type=str, default='FracTALResNet',
                    help='Model base for feature extraction, default::FracTALResNet')
parser.add_argument('--depth',type=int, default=3,
                    help='XX depth, default::3')
parser.add_argument('--widths',type=list, default=[2],
                    help='XX widths, default::2')
parser.add_argument('--nheads_start',type=int, default=32//8,
                    help='XX nheads_start, default::{}'.format(32//8))
parser.add_argument('--in_features',type=int, default=1024,
                    help='XX in_features, default::{}'.format(1024))
parser.add_argument('--causal',type=bool, default=True,
                    help='Model predictions HEAD, causal vs standard, default:: causal=True')


opt = parser.parse_args()


import sys
sys.path.append(opt.root)

# Data augmentation definitions 
from mxnet.gluon.data.vision import transforms

# Model definitions 
from GalaxyZoo.mxnet.models.xxnets import * 

if opt.causal:
    tname = r'causal'
flname_write = r'Results/'+ opt.model + r'_' + tname +r'.txt' 

# ================== SAVING best model ==================================
import datetime, os
stamp = datetime.datetime.now().strftime('%Y-%m-%d-Time-%H:%M:%S_') 
flname_save_weights = r'Results/' + stamp + opt.model+ '_best_model.params'
# =========================================================================

# Decide on cuda: 
if opt.cuda and mx.util.get_gpu_count():
    ctx = [mx.gpu(i) for i in range(mx.util.get_gpu_count())]
else:
    ctx = [mx.cpu()]

# Define model
net = xxnets(nfilters_init = opt.nfilters_init, depth=opt.depth, widths=opt.widths, nheads_start = opt.nheads_start, model_name = opt.model, causal = opt.causal, psp_depth=2)
net.initialize(ctx=ctx)
net.hybridize()  # ZoomZoom!! 


# Data augmentation definitions 
transform_train = transforms.Compose([
    # Randomly crop an area, and then resize it to be 32x32
    transforms.RandomResizedCrop(opt.crop_size,scale=(0.2,1.)),# test also with 0.6
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
    transforms.Resize(opt.crop_size),
    transforms.ToTensor(),
    transforms.Normalize([11.663384, 10.260227,  7.65015 ], [21.421959, 18.044296, 15.494861])
])

# Datasets/DataLoaders 
from GalaxyZoo.mxnet.src.GZooDataset import *
dataset_train = GZooData(root=os.path.join(opt.root,'GalaxyZoo'), transform=transform_train)
datagen_train = gluon.data.DataLoader(dataset_train,batch_size=opt.batch_size,shuffle=True,last_batch='rollover')

dataset_dev = GZooData(root=os.path.join(opt.root,'GalaxyZoo'), mode='dev',transform=transform_test)
datagen_dev = gluon.data.DataLoader(dataset_dev,batch_size=opt.batch_size,shuffle=False,last_batch='rollover')


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



train_metric = gluon.metric.MSE()
loss_fn  = gluon.loss.L1Loss() 

# development metric: 
def test(tctx, tnet, tdatagen_dev):
    metric = gluon.metric.MSE()
    print ("started testing ...")
    for idx, data in enumerate(tdatagen_dev):
        print("\rRunning:: {}/{}".format(idx+1,len(tdatagen_dev)),end='',flush=True)
        #data = gluon.utils.split_and_load(batch[0], ctx_list=ctx, batch_axis=0)
        imgs, labels = data
        #imgs = imgs.as_in_context(tctx)
        imgs = gluon.utils.split_and_load(imgs, ctx_list=ctx, batch_axis=0)
        #outputs = nd.concatenate(outputs,axis=0)
        with mx.autograd.predict_mode():
            preds = [tnet(timgs).as_in_context(mx.cpu()) for timgs in imgs]
        preds = mx.np.concatenate(preds,axis=0)
        metric.update(preds=preds, labels=labels)
        mx.npx.waitall() # necessary to avoid memory flooding 
    return metric.get()

def train(epochs,ctx,flname_write):

    with open(flname_write,"w") as f:
        print('epoch','train_mse','val_mse','train_loss',file=f,flush=True)

        ref_metric = 1000
        for epoch in range(epochs):
            tic = time.time()
            train_metric.reset()
            train_loss = 0

            # Loop through each batch of training data
            for i, (data,label) in enumerate(datagen_train):
                print("\rWithin epoch completion:: {}/{}".format(i+1,len(datagen_train)),end='',flush=True)
                # Extract data and label
                data = gluon.utils.split_and_load(data,ctx_list=ctx)
                label = gluon.utils.split_and_load(label,ctx_list=ctx)

                # AutoGrad
                with autograd.record():
                    outputs = [net(tdata) for tdata in data]
                    losses  = [loss_fn(tout,tlabel).mean() for tout, tlabel in zip(outputs,label)]
                    
                    # necessary to avoid memory flooding 
                    mx.npx.waitall() 
                    # Backpropagation
                    for l in losses:
                        l.backward()


                # Optimize
                trainer.step(opt.batch_size) # This is the batch_size

                
                # Update metrics
                train_loss += sum(losses)
                train_metric.update(label, outputs)
            train_loss = train_loss / len(datagen_train)
            name, train_mse = train_metric.get()
            # Evaluate on Validation data
            nd.waitall() # necessary to avoid cuda malloc
            name, val_mse = test(ctx, net, datagen_dev)

            # Print metrics
            # print both on screen and in file 
            print("\n")
            print('epoch={} train_mse={} val_mse={} train_loss={} time={}'.format(epoch, train_mse, val_mse, train_loss, time.time()-tic))
            print(epoch, train_mse, val_mse, train_loss, file=f,flush=True)
            if val_mse < ref_metric:
                # Save best model parameters, according to minimum val_mse
                net.save_parameters(flname_save_weights)
                ref_metric = val_mse



if __name__=='__main__':
    #tout = test(ctx,net,datagen_dev)
    #print ("Passed first test")
    train(opt.epochs, ctx, flname_write)
