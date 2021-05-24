import numpy as np
import cv2 as cv

import convolution as conv


def read_pgm2(pgm_file):
    """Return raster of image from file P2 .pgm"""
    assert pgm_file.readline() != 'P2\n'

    (width, height) = [int(i) for i in pgm_file.readline().split()]
    max_value = int(pgm_file.readline())

    raster = []
    for _ in range(height):
        row = [int((int(i)/max_value)*255) for i in pgm_file.readline().split()]
        raster.append(row)

    return np.array(raster).astype('uint8')


def addArrows(image, scale, relo_matrix):
    """It is only for test"""
    for relo in relo_matrix:
        a = int(relo[4])
        cv.arrowedLine(image,(relo[1]*scale,relo[0]*scale),(relo[3]*scale,relo[2]*scale),(0,a,0),2,tipLength = 0.1)
    return image


def prepareImage(image, relo_matrix):
    scale = 4
    img_colored = cv.applyColorMap(image, cv.COLORMAP_MAGMA)
    img_resized = cv.resize(img_colored,
                            (img_colored.shape[1]*scale, img_colored.shape[0]*scale),
                            interpolation=cv.INTER_CUBIC)
    img_arrowed = addArrows(img_resized, scale, relo_matrix)
    return img_arrowed

def prepareImage2(image):
    scale = 4
    img_colored = cv.applyColorMap(image, cv.COLORMAP_MAGMA)
    img_resized = cv.resize(img_colored,
                            (img_colored.shape[1]*scale, img_colored.shape[0]*scale),
                            interpolation=cv.INTER_CUBIC)
    return img_resized

def readAllImage(images, n):
    for i in range(n):
        f = open("image_video/IMG_"+f"{i:04}"+".pgm", 'rb')
        images[i,:,:] = read_pgm2(f)
        f.close()


def main():
    #read image
    f = open("image/IMG_0006.pgm", 'rb')
    image1 = read_pgm2(f)
    f.close()
    f = open("image/IMG_0002.pgm", 'rb')
    image2 = read_pgm2(f)
    f.close()


    con = conv.BlobsRelocation()
    con.prepareSettingsFromImage(image1)
    res_matrix = con.calculateRelocation(image1, image1)

    abc = prepareImage(image1, res_matrix)
    abc2 = prepareImage2(image1)

    img = np.append(abc, abc2, axis=1)
    cv.imshow("image", img)
    cv.waitKey()


if __name__ == '__main__':
    main()
