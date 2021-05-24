import numpy as np
import cv2 as cv


class BlobsRelocation:
    def __init__(self, blob_size=21, range_to_search=100, corr_step=3):
        # Image data
        self.image_rows = None
        self.image_cols = None

        # Data for corr
        self.range_to_search = range_to_search
        self.blob_size = blob_size
        self.corr_step = corr_step
        self.blobs_position = None

    def calculateRelocation(self, current_image, previous_image):
        """
        Function calculate max value of phase corr for all blobs.
        :param current_image:
        :param previous_image:
        :return: vector n x 5
        """

        # 5 is count of params row,col,new_row,new_col, corr
        blobs_shift = np.zeros((len(self.blobs_position), 5)).astype("int8")
        for i in range(len(self.blobs_position)):
            max_corr = self.findMaximalFit(current_image, previous_image, self.blobs_position[i])
            blobs_shift[i][0] = self.blobs_position[i][0]
            blobs_shift[i][1] = self.blobs_position[i][1]
            blobs_shift[i][2] = max_corr[0]
            blobs_shift[i][3] = max_corr[1]
            blobs_shift[i][4] = max_corr[2]

        return blobs_shift


    def findMaximalFit(self, current_image, previous_image, blob_position):
        """
        Function try find best fit blobs from preview image to current.
        If max_range=None - function try find in all image.
        :param previous_image:
        :param current_image: image to fit
        :param blob_position: position blob to fit
        :return:
        """

        current_image = current_image.astype(float)
        mask_with_blob = np.zeros((self.image_rows, self.image_cols))

        mask_with_blob[0:self.blob_size, 0:self.blob_size] = \
            previous_image[blob_position[0]:blob_position[0] + self.blob_size,
            blob_position[1]:blob_position[1] + self.blob_size]

        corr = cv.phaseCorrelate(mask_with_blob, current_image)

        max_corr_int8 = np.array([corr[0][1],
                                  corr[0][0],
                                  np.round(corr[1] / (1. / np.iinfo("int8").max))]).astype("int8")
        return max_corr_int8


    def prepareBlobsPosition(self):
        """
        Method create vector of position blobs.
        Position its a up left corner
        :return:
        """

        # Calculate offset to get blobs the closest center
        first_blob_row = (self.image_rows % self.blob_size) // 2
        first_blob_col = (self.image_cols % self.blob_size) // 2

        blobs_position_row = np.arange(first_blob_row, self.image_rows - self.blob_size + 1, self.blob_size)
        blobs_position_col = np.arange(first_blob_col, self.image_cols - self.blob_size + 1, self.blob_size)

        self.blobs_position = [[row, col] for row in blobs_position_row for col in blobs_position_col]

    def prepareSettingsFromImage(self, image):
        """
        Function must be used when first image come
        :param image: first image
        :return:
        """
        self.image_rows, self.image_cols = image.shape[:2]
        self.prepareBlobsPosition()
