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


def draw_bbox_heatmap(img, bboxes, heatmap):
    bbox_img=draw_bboxes(img, bboxes, show=False)
    fig=plt.figure(figsize=(10,10))
    fig.add_subplot(1,2,1)
    plt.imshow(bbox_img)
    fig.add_subplot(1,2,2)
    plt.imshow(heatmap)
    plt.show()
    return bbox_img