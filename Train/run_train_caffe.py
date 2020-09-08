#!/usr/bin/env python

caffe_root = 'crfasrnn/caffe/'
import sys, getopt
sys.path.insert(0, caffe_root + 'python')
import caffe
#import surgery, score
import subprocess
import numpy as np
import os



# make a bilinear interpolation kernel
# credit @longjon
def upsample_filt(size):
    factor = (size + 1) // 2
    if size % 2 == 1:
        center = factor - 1
    else:
        center = factor - 0.5
    og = np.ogrid[:size, :size]
    return (1 - abs(og[0] - center) / factor) * \
           (1 - abs(og[1] - center) / factor)

# set parameters s.t. deconvolutional layers compute bilinear interpolation
# N.B. this is for deconvolution without groups
def interp_surgery(net, layers):
    for l in layers:
        print(l)
        m, k, h, w = net.params[l][0].data.shape
        #print(m,k,h,w)
        if m != k:
            print 'input + output channels need to be the same'
            raise ValueError('go away') 
        if h != w:
            print 'filters need to be square'
            raise  ValueError('no away')

        filt = upsample_filt(h)
        net.params[l][0].data[range(m), range(k), :, :] = filt


#weights = 'train_iter_200000.caffemodel'
weights ='train_32_classes_iter_1000.caffemodel'
solver = caffe.SGDSolver('solver.prototxt')
#solver.net.copy_from(weights)

#interp_layers = [k for k in solver.net.params.keys() if 'up' in k]

# init
caffe.set_device(0)
caffe.set_mode_gpu()
print('cA')
#solver = caffe.SGDSolver('solver_2.prototxt')
solver.net.copy_from(weights)
for layer in solver.net.params.keys():
  for index in range(0, 2):
    if len(solver.net.params[layer]) < index+1:
      continue
   # print(solver.net.params[layer][index])
    if np.sum(solver.net.params[layer][index].data) == 0:
      print layer + ' is composed of zeros!'
      halt_training = True
# surgeries
#interp_layers = [k for k in solver.net.params.keys() if 'up' in k]
#interp_layers = [k for k in solver.net.params.keys() if 'up' in k or 'score2' in k or 'score4' in k or 'score59' in k]
#surgery_interp(solver.net, interp_layers)

# scoring


solver.step(1000000)
