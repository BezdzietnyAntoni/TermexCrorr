import numpy as np


def read_pgm2(pgm_path):
    """Return raster of image from file P2 .pgm"""
    pgm_file = open(pgm_path, 'rb')

    assert pgm_file.readline() != 'P2\n'

    (width, height) = [int(i) for i in pgm_file.readline().split()]
    max_value = int(pgm_file.readline())

    raster = []
    for _ in range(height):
        row = [int((int(i) / max_value) * 255) for i in pgm_file.readline().split()]
        raster.append(row)

    pgm_file.close()

    return np.array(raster).astype('uint16')