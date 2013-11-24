import matplotlib.pyplot as plt
import numpy as np
import Image
from skimage import data
from skimage.data import camera, coins
from skimage import filter
import skimage.io as io
from skimage.measure import structural_similarity as ssim


'''
mean square error
'''
def mse(x, y):
    return np.linalg.norm(x - y)




#get first reference image
refImage1 = Image.open("ref1.png")
#convert to grayscale
refImage1 = refImage1.convert("L")
refImage1.load()
refImage1NPArray = np.asarray(refImage1, dtype=np.float64)

#second reference image
refImage2 = Image.open("ref2.png")
#convert to grayscale
refImage2 = refImage2.convert("L")
refImage2.load()
refImage2NPArray = np.asarray(refImage2, dtype=np.float64)

#first test image
testImage1 = Image.open("test1.png")
#convert to grayscale
testImage1 = testImage1.convert("L")
testImage1.load()
testImage1NPArray = np.asarray(testImage1, dtype=np.float64)

#second test image
testImage2 = Image.open("test2.png")
#convert to grayscale
testImage2 = testImage2.convert("L")
testImage2.load()
testImage2NPArray = np.asarray(testImage2, dtype=np.float64)

#similarity scores
mse_test1_ref1 = mse(refImage1NPArray, testImage1NPArray)
mse_test1_ref2 = mse(refImage2NPArray, testImage1NPArray)
mse_test2_ref1 = mse(refImage1NPArray, testImage2NPArray)
mse_test2_ref2 = mse(refImage2NPArray, testImage2NPArray)


ssim_test1_ref1 = ssim(refImage1NPArray, testImage1NPArray)
ssim_test1_ref2 = ssim(refImage2NPArray, testImage1NPArray)
ssim_test2_ref1 = ssim(refImage1NPArray, testImage2NPArray)
ssim_test2_ref2 = ssim(refImage2NPArray, testImage2NPArray)



fig, (ax0, ax1, ax2, ax3) = plt.subplots(ncols=4)
print ax0
print ax1
print ax2
print ax3

label = 's1: %.2f, s2: %.2f'
#get it on the plot
ax0.imshow(refImage1NPArray, cmap=plt.cm.gray)
ax1.imshow(refImage2NPArray, cmap=plt.cm.gray)
ax2.imshow(testImage1NPArray, cmap=plt.cm.gray)
ax2.set_xlabel(label % (ssim_test1_ref1, ssim_test1_ref2))
ax3.imshow(testImage2NPArray, cmap=plt.cm.gray)
ax3.set_xlabel(label % (ssim_test2_ref1, ssim_test2_ref2))
#display plot

plt.show()



'''
image = Image.open("ref1.png")
image = image.convert("L")
image.load()
imgd = np.asarray(image, dtype=np.float64)
print imgd
io.imshow(imgd)
print image.format, image.size, image.mode
print imgd.shape


coinsimg = data.coins()
print coinsimg.shape
image = np.fromfile("test1.png", dtype=np.int64)
print image.shape

#image = camera()
edge_roberts = filter.canny(imgd)
edge_sobel = filter.roberts(imgd)
#edge_sobel = filter.threshold_otsu(coinsimg)
#io.imshow(image)
fig, (ax0, ax1) = plt.subplots(ncols=2)
print ax0
print ax1

ax0.imshow(edge_roberts, cmap=plt.cm.gray)
ax0.set_title('Roberts Edge Detection')
ax0.axis('off')

ax1.imshow(edge_sobel, cmap=plt.cm.gray)
ax1.set_title('Sobel Edge Detection')
ax1.axis('off')

plt.show()

'''