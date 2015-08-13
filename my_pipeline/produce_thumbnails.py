import scipy.io as sio
import scipy.misc
import numpy as np
import pylab as pl
import Image
import os, sys
import registration as reg
mat_contents = sio.loadmat('./data/testImgInfo.mat')
# print mat_contents
imgDIR = mat_contents['imgDIR']
# print imgDIR.shape
imgNames = mat_contents['imgNames']
# print imgNames.shape
imgPos = mat_contents['imgPos']
# print imgPos.shape
width = 128
height = 64
stage = 2
orientation = 'lateral'
cvTerm = 'all'
storedir = './thumbnails/StageBin' + str(stage) + '/' + orientation + '/' + cvTerm + '/'
# mkdir([store_dir,'images/']);

template = reg.generateTemplate(width / 2, height / 2)
# print template.shape
content = os.listdir('./data/rawImg/')
content_length = len(content)
# print content_length
for i in xrange(0, content_length):
    fileName = str(imgDIR[i][0][0]);
    # print fileName
    fileName2 = reg.myinsitu2fn(fileName, 1);
    fileName2 = fileName2.replace('jpe', 'jpg')
    # print fileName2
    if len(fileName) > 0:
        # matplotlib can only read PNGs natively, but if PIL is installed, it will use it to load the image and return an array (if possible) which can be used with imshow().
        url = './data/rawImg/' + fileName2
        img = Image.open(url)
        I = np.asarray(img)
        # print "Before: " + str(I.shape)
        # pl.imshow(I)
        # pl.show()
        (m, n, k) = I.shape
        fac = 900. / m
        I = scipy.misc.imresize(I, fac)
        # print "After: " + str(I.shape)
        pl.imshow(I)
        pl.show()

    # Step 1: Image segmentation
