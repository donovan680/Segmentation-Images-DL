import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import random 
import dircache
import os
from random import randint
import string
import oct2py 
oc = oct2py.Oct2Py()


alpha1 = 0

def to_rgb1(im):
    # I think this will be slow
    for i in range(im.shape[0]):
       for j in range(im.shape[1]):
          for k in range(im.shape[2]):
              if im[i,j,k]==0:
                 im[i,j,k]==255
    return im


def give_name(i):
   if (i==0):
     name='backgournd'
   elif i==1:
     name='hat'
   elif i==2:
     name='hair'
   elif i==3:
     name='eyes'
   elif i==4:
      name='Robe-Tshirt'
   elif i==5:
      name='short skirt'
   elif i==6:
      name='trousers'
   elif i==7:
      name='skirt'
   elif i==8:
      name='belt'
   elif i==9:
      name='shoes'
   elif i==10:
      name='shoes'
   elif i==11:      
     name='Face'
   elif i==12:      
     name=' left leg'
   elif i==13:      
     name=' right leg'
   elif i==14:    
     name=' Right Hand'
   elif i==15:      
     name=' left Hand'
   elif i==16:
      name='handbag'
   else:
      name='17'
   return name 
      
   
dir = 'Results'
filename = random.choice(dircache.listdir(dir))
path = os.path.join(dir, filename)
print(filename)
#mask
mm=Image.open(path)
#mm.show()
mask = np.array(Image.open(path)) #mask given by caffe
A=np.unique(mask)
x = oc.imread(path );oc.figure(1);oc.imagesc(x);oc.axis('off');oc.saveas(1,'t.png');


#image
image_dir=string.replace(filename, '_segmentation.png', '.jpg')
image_name='SH_IMG/'+image_dir
img = np.array(Image.open(image_name))#image used as test 
chk = 255.0*np.ones(img.shape,dtype=img.dtype)


# all the work done in these two lines
mask = mask[:,:,np.newaxis]

print(np.unique(chk))
#res=give_white(res)

print(A)

#result.show()
imge=Image.fromarray(img)
fig = plt.figure()
X=3;Y=5
number_candiates=X*Y
plt.subplot(X,Y,3)
plt.imshow(imge)
plt.axis('off')
plt.subplot(X,Y,2)
e=Image.open('t.png')
t = e.resize([300, 300], Image.ANTIALIAS)
plt.imshow(t)
plt.axis('off')

for i in range(len(A)):
    mon_elemnt=A[i]
    res = np.where(mask!=mon_elemnt, chk, img)
    res=np.array(res,dtype=img.dtype)
    #print(np.unique(res))

    result = Image.fromarray(res)
    plt.subplot(X,Y,Y+i+1)
    
    plt.imshow(res)
    plt.axis('off')
    ax1 = fig.add_subplot(X,Y,Y+i+1)
    ax1.title.set_text(give_name(A[i]))
save_name= 'my_doc/'+filename
plt.savefig(save_name)
plt.show()
print(res.shape)

#result.save(str(mon_elemnt)+'_seg.jpg')


