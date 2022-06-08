import os
import random
from operator import itemgetter

import numpy as np
from PIL import Image
from .base_dataset import BaseDataSet

class ClothesDataset(BaseDataSet):

    def __init__(self, **kwargs):
        self.classes = ['top', 'skirt', 'outer', 'dress', 'bottom']
        self.num_classes = len(self.classes) + 1
        super(ClothesDataset, self).__init__(**kwargs)

    def _set_files(self):
        
        self.image_files = [os.listdir(os.path.join(self.root,i,'image')) for i in self.classes]
        for i,v in enumerate(self.image_files) : 
            self.image_files[i] = [os.path.join(self.root,self.classes[i],'image',j) for j in v]
        self.image_files = sum(self.image_files,[])
        random.shuffle(self.image_files)
                                      
        self.label_files = [(i.split('.')[0]+'-label-preview.png').replace('image','label') for i in self.image_files]

    def get_classes(self):
        t = {self.classes[i]:i+1 for i in range(len(self.classes))}
        t['background'] = 0
        return dict(sorted(t.items(), key=itemgetter(1)))
    
    def get_cmap(self):
        t = Image.open(self.label_files[0])
        return t.getpalette()

    def _load_data(self, index):
        image_path = self.image_files[index]
        label_path = self.label_files[index]
        
        image = np.asarray(Image.open(image_path), dtype=np.float32)
        if image.shape[2] == 4 :
            image = image[:,:,:3]
        label = np.asarray(Image.open(label_path), dtype=np.int32)
        
        return image, label, image_path
    
    
    
    
    
    