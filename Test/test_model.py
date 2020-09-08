import numpy as np

from PIL import Image

import sys

sys.path.append("caffe/python")
import caffe 
#caffe.set_mode_gpu()
def mse(imageA, imageB):
	# the 'Mean Squared Error' between the two images is the
	# sum of the squared difference between the two images;
	# NOTE: the two images must have the same dimension
	err = np.sum((imageA.astype("float") - imageB.astype("float")) ** 2)
	err /= float(imageA.shape[0] * imageA.shape[1])
	
	# return the MSE, the lower the error, the more "similar"
	# the two images are
	return err

# load image, switch to BGR, subtract mean, and make dims C x H x W for Caffe

im = Image.open('a.jpg')

#im = im.resize([300, 300], Image.ANTIALIAS)

in_ = np.array(im, dtype=np.float32)

in_ = in_[:,:,::-1]

in_ -= np.array((104.00698793,116.66876762,122.67891434))

in_ = in_.transpose((2,0,1))



# load net

# net = caffe.Net('deploy.prototxt', 'train_iter_100000.caffemodel', caffe.TEST)

#pretrained_net='train_segmentation_Cloths_iter_150000.caffemodel'
pretrained_net='train_32_classes_iter_100000.caffemodel'


net = caffe.Net('deploy_32.prototxt', pretrained_net, caffe.TEST)



# shape for input (data blob is N x C x H x W), set data

net.blobs['data'].reshape(1, *in_.shape)

net.blobs['data'].data[...] = in_

# run net and take argmax for prediction

net.forward()

out = net.blobs['score-a'].data[0].argmax(axis=0)

out = np.uint8(out)
print('ou')
print(out.shape)
result = Image.fromarray(out)

print(np.unique(out))


  #print(array_image.shape)
#x=out
x =out *255
#plt.imshow(x, cmap='gray', interpolation='nearest', vmin=0, vmax=255)

  # Creates PIL image
img = Image.fromarray(x, 'L')
  #img.show()
  #print(np.unique(array_image))
  #Rescale to 0-255 and convert to uint8
data=x
rescaled = (255.0*data).astype(np.uint8)
  
im = Image.fromarray(rescaled)
save_image='a.png'
im.save(save_image)
  #im = Image.open('test.png')
  #in_ = np.array(im).astype('uint8')
  #print(np.unique(in_))
