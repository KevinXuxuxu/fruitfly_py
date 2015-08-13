import numpy as np
import math

def generateTemplate(a, b, c = 3):
    assert(a >= b)
    templateImage = np.zeros((2 * b, 2 * a, c))
    for j in xrange(1, a+1):
        leng = math.floor(b * math.sqrt(1 - ((j - 0.5) / a)**2) + 0.5)
        # print leng
        if leng > 0:
            templateImage[1-leng+b-1: leng+b, 1-j+a-1: j+a, ...] = 1
            # print str(int(1-leng+b)) + ' ' + str(int(leng+b)) + ' ' + str(1-j+a) + ' ' + str(j+a)
    return templateImage

# test generateTemplate
# if __name__ == "__main__":
#     test = generateTemplate(64, 32, 3)
#     import scipy.io as sio
#     mat = sio.loadmat('generateTemplate_test.mat')
#     ans = mat['generateTemplate_test']
#     if (test == ans).all():
#         print "Test Success!"

def myinsitu2fn(filename, to_emesh_inp):
    # [myquery] = myinsitu2fn(filename, to_emesh_inp)
    # Converts filenames between insitu MySQL database and emesh
    # to_emesh_inp can be omitted
    # If to_emesh_inp is given and not equal 0, filename will be converted to
    # enmesh compatible filename
    # Otherwise filename will be converted from emesh to MySQL query compatible
    # format
    #
    # Convert filename to emesh compatible
    fn_idx = filename.find('/')
    return filename[fn_idx+1 : ]

    # test myinsitu2fn
    # if __name__ == "__main__":
    # 	fileName = "img_dir_21/insitu21077.jpe"
    # 	fileName2 = myinsitu2fn(fileName, 1)
    # 	print fileName2