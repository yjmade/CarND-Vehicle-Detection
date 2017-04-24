# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import cv2


def draw_bboxes(img, bboxes, show=True):
    # Iterate through all detected cars
    for xmin, ymin, xmax, ymax in bboxes:
        cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (0, 0, 255), 6)
    if show:
        plt.imshow(img)
        plt.show()
    return img
