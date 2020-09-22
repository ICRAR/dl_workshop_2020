from PIL import Image
from mxnet import gluon, nd
import pandas as pd 
import os

class GZooData(gluon.data.Dataset):
    def __init__(self, root = r'/Users/foivos/Documents/kaggle_comps/GalaxyZoo/data/',mode = 'train',  transform=None):
        
        self._transform = transform
               
        self.root_imgs = os.path.join(root ,'data','images_training_rev1')
        self.root_data = os.path.join(root,'data')
        self.df = pd.read_csv( os.path.join(self.root_data, 'training_solutions_rev1.csv'),sep=",")
 

        if mode=='train':
            self.idxs = pd.read_csv(os.path.join(self.root_data,'train_indices.csv'))
        elif mode =='dev':
            self.idxs = pd.read_csv(os.path.join(self.root_data,'dev_indices.csv'))
        elif mode == 'test':
            self.idxs = pd.read_csv(os.path.join(self.root_data,'test_indices.csv'))
        else:
            raise ValueError("I do not understand given mode::{}, aborting ...".format(mode))
            
        # Get the FIXED split 
        self.df = self.df[self.df.GalaxyID.isin(self.idxs.GalaxyID)]
        
    def __getitem__(self, idx):
        
        GID = int(self.df.iloc[idx]['GalaxyID'])
        probs = self.df.iloc[idx,1:].to_numpy()
        probs = nd.array(probs)
        
        path_read = os.path.join(self.root_imgs , str(GID)+r'.jpg')
        
        img = Image.open(path_read)
        img = nd.array(img)
        
        if self._transform is not None:
            img = self._transform(img)

        return img, probs
    
    def __len__(self):
        return len(self.df)

