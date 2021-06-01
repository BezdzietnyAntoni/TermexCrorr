import numpy as np
import cv2 as cv
from pgmReader import read_pgm2
from termexCor import TermexCorr
from termexVis import TermexVis


def main():
    blob_size = 44
    distance_between_blobs = (10, 12)
    corr = TermexCorr(blob_size, distance_between_blobs)
    vis = TermexVis(scale=5, blob_size=blob_size)

    last_image = read_pgm2("image_video/IMG_0130.pgm")
    corr.prepareCorrelationSetting(last_image)

    idx = 130
    quit = False
    while not quit:
        idx = (idx + 1) % 220
        idx = 130 if idx < 130 else idx
        image = read_pgm2("image_video/IMG_" + f"{idx:04}" + ".pgm")

        corr_arrow = corr.calculateRelocation(image, last_image)

        prev_image = vis.prepareImage(last_image.astype("uint8"))
        curr_image = vis.prepareImageWithArrow(image.astype("uint8"), corr_arrow, 10)

        connected_image = np.append(prev_image, curr_image, axis=1)

        last_image = image[:]

        cv.imshow("Termex", connected_image)
        keyCode = cv.waitKey(500)
        if keyCode > 0:
            cv.destroyAllWindows()
            quit = True


if __name__ == '__main__':
    main()
