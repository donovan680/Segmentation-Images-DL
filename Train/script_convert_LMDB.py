

import sys 
sys.path.append("caffe/python")
import caffe
import lmdb
from PIL import Image
import numpy as np
import glob
from random import shuffle

import h5py


##data form two folder IMG: image and AN: anotation
inputs_data = sorted(glob.glob("IMG/*.jpg"))

inputs_label = sorted(glob.glob("AN/*.png"))


N=len(inputs_data)
print(N)
NumberTrain=N
NumberTest=int(N)
inputs_Train = inputs_data[0:5000] # Extract the training data from the complete set
inputs_Train_Label=inputs_label[0:5000]
inputs_Test = inputs_data[5000:7000] # Extract the testing data from the complete set
inputs_Test_Label=inputs_label[5000:7000]


print("Creating Training Data LMDB File ..... ")
in_db = lmdb.open('Train_Data_lmdb',map_size=int(1e12))
with in_db.begin(write=True) as in_txn:
	for in_idx, in_ in enumerate(inputs_Train):
	   # print in_[-8:-3]
	    print in_idx,in_
	    im = np.array(Image.open(in_)) # or load whatever ndarray you need
	    Dtype = im.dtype
            if (len(im.shape)==3):
	       im = im[:,:,::-1]             
	            
               im = im.transpose((2,0,1))
            else:
               im = im.astype(np.uint8)
               
               im = im[np.newaxis, :, :] 
	    im_dat = caffe.io.array_to_datum(im)
	    in_txn.put('{:0>10d}'.format(in_idx),im_dat.SerializeToString())

in_db.close()
print("Creating Training Label LMDB File ..... ")

in_db = lmdb.open('Train_Label_lmdb',map_size=int(1e12))

with in_db.begin(write=True) as in_txn:
	for in_idx, in_ in enumerate(inputs_Train_Label):
	    print in_idx,in_   
            im=np.array(Image.open(in_))
	    im = im.astype(np.uint8)
            im = im[np.newaxis, :, :]
 	    im_dat = caffe.io.array_to_datum(np.array(im))
	    in_txn.put('{:0>10d}'.format(in_idx),im_dat.SerializeToString())
	    
in_db.close()
print 'ok'

print('test_lambd')
in_db = lmdb.open('Test_Data_lmdb',map_size=int(1e12))
with in_db.begin(write=True) as in_txn:
        for in_idx, in_ in enumerate(inputs_Test):
            print in_[-8:-3]
            print in_idx,in_
            im = np.array(Image.open(in_)) # or load whatever ndarray you need
            Dtype = im.dtype
            if (len(im.shape)==3):
               im = im[:,:,::-1]  
               im = im.transpose((2,0,1))
            else:
               im = im.astype(np.uint8)               
               im = im[np.newaxis, :, :] 
            im_dat = caffe.io.array_to_datum(im)
            in_txn.put('{:0>10d}'.format(in_idx),im_dat.SerializeToString())

in_db.close()

print("Creating Testing Label LMDB File ..... ")

in_db = lmdb.open('Test_Label_lmdb',map_size=int(1e12))

with in_db.begin(write=True) as in_txn:
        for in_idx, in_ in enumerate(inputs_Test_Label):
            print in_idx          
            im=np.array(Image.open(in_))
            im = im.astype(np.uint8)          
            im = im[np.newaxis, :, :]
            im_dat = caffe.io.array_to_datum(np.array(im))
            in_txn.put('{:0>10d}'.format(in_idx),im_dat.SerializeToString())

in_db.close()

