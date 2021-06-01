import numpy as np
import cv2 as cv


class TermexVis:
    def __init__(self, scale=5, blob_size=40):
        self.scale = scale
        self.blob_size = blob_size

    def resizeImage(self, image):
        image_resized = cv.resize(image,
                                  (image.shape[1] * self.scale,
                                   image.shape[0] * self.scale),
                                  interpolation=cv.INTER_CUBIC)

        return image_resized

    @staticmethod
    def addColorMap(image):
        image_colored = cv.applyColorMap(image.astype("uint8"), cv.COLORMAP_MAGMA)
        return image_colored

    def addArrows(self, image, arrows_vector, threshold):
        for arrow in arrows_vector:
            factor = int(arrow[4])
            if factor < threshold:
                break

            start_point = ((arrow[1] + self.blob_size//2) * self.scale,
                           (arrow[0] + self.blob_size//2) * self.scale)
            end_point = ((arrow[3] + self.blob_size//2) * self.scale,
                         (arrow[2] + self.blob_size//2) * self.scale)

            if (image.shape[1] < end_point[0]) or (end_point[0] < 0):
                break

            if (image.shape[0] < end_point[1]) or (end_point[1] < 0):
                break

            cv.arrowedLine(image, start_point, end_point, (0, 50+2*factor, 0), 2)

        return image

    def prepareImage(self, image):
        resized_image = self.resizeImage(image)
        resized_colored_image = self.addColorMap(resized_image)
        return resized_colored_image

    def prepareImageWithArrow(self, image, arrows_vector, threshold=5):
        resized_image = self.resizeImage(image)
        resized_colored_image = self.addColorMap(resized_image)
        arrowed_image = self.addArrows(resized_colored_image, arrows_vector, threshold)
        return arrowed_image
