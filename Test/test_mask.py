
from PIL import Image
import numpy as np
mask = np.array(Image.open('a.png'))
img = np.array(Image.open('a.jpg'))
all_class=np.unique(mask)
random_class=int(random.choice(all_class))#example: Shoes (16)
chk = np.ones(img.shape,dtype=img.dtype)
# all the work done in these two lines
mask = mask[:,:,np.newaxis]
res = np.where(mask==random_class, chk, img)
res=np.array(res,dtype=img.dtype)
print(img.shape)
print(mask.shape)
print(res.shape)
result = Image.fromarray(res)
result.show()
result.save('image_with_segmentation.jpg')
