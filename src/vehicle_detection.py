# -*- coding: utf-8 -*-
import cv2
import numpy as np
from scipy.ndimage.measurements import label as sk_label
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(1, os.path.dirname(os.path.dirname(__file__)))


from train_svm import get_model, imread, imwrite
from visualization import draw_bboxes


model = get_model()


def slide_window(
    img_shape,
    x_start_stop=[None, None],
    y_start_stop=[None, None],
    xy_window=(64, 64),
    xy_overlap=(0.5, 0.5)
):
    # If x and/or y start/stop positions not defined, set to image size
    # Compute the span of the region to be searched
    # Compute the number of pixels per step in x/y
    # Compute the number of windows in x/y
    # Initialize a list to append window positions to
    h, w = img_shape[:2]
    x_start, x_stop = x_start_stop
    y_start, y_stop = y_start_stop
    x_start = x_start or 0
    x_stop = x_stop or w
    y_start = y_start or 0
    y_stop = y_stop or h
    x_overlap, y_overlap = xy_overlap
    w_window, h_window = xy_window

    return [
        (xmin, ymin, xmin + w_window, ymin + h_window)
        for xmin in range(x_start, x_stop, int((1 - x_overlap) * w_window))
        for ymin in range(y_start, y_stop, int((1 - y_overlap) * h_window))
        if xmin + w_window <= x_stop and ymin + h_window <= y_stop
    ]
    # Loop through finding x and y window positions
    #     Note: you could vectorize this step, but in practice
    #     you'll be considering windows one by one with your
    #     classifier, so looping makes sense
    # Calculate each window position
    # Append window position to list
    # Return the list of windows


def add_heat(heatmap, bbox):
    # Iterate through list of bboxes
    xmin, ymin, xmax, ymax = bbox
    # Add += 1 for all pixels inside each bbox
    # Assuming each "box" takes the form ((x1, y1), (x2, y2))

    heatmap[ymin:ymax, xmin:xmax] += 1

    # Return updated heatmap
    return heatmap


def apply_threshold(heatmap, threshold):
    # Zero out pixels below the threshold
    heatmap[heatmap <= threshold] = 0
    # Return thresholded map
    return heatmap


def batch_loop(iter_, batch_size, item_func=lambda x: x):
    iter_ = iter(iter_)
    for item in iter_:
        item = item_func(item)
        if isinstance(item, (tuple, list)):
            item_len = len(item)
            batch_list_gen = lambda: [[] for _ in range(item_len)]
            batch_list_append = lambda batch_list, items: [batch.append(item) for batch, item in zip(batch_list, items)]
        else:
            batch_list_gen = list
            batch_list_append = lambda batch_list, item: batch_list.append(item)

        batch_list = batch_list_gen()
        batch_list_append(batch_list, item)
        break
    else:
        return
    for i, item in enumerate(iter_, 1):
        if i % batch_size == 0:
            yield batch_list
            batch_list = batch_list_gen()
        batch_list_append(batch_list, item_func(item))

    yield batch_list


def slide_window_predict(img, batch_size, **kwargs):
    # for window_batch, img_batch in batch_loop(
    #     slide_window(img, **kwargs),
    #     batch_size,
    #     item_func=lambda window_bbox: (window_bbox, img[window_bbox[1]:window_bbox[3], window_bbox[0]:window_bbox[2]])
    # ):
    windows = slide_window(
        img.shape,
        **kwargs
    )
    window_imgs = [
        cv2.resize(
            img[window_bbox[1]:window_bbox[3], window_bbox[0]:window_bbox[2]],
            (64, 64)
        )
        for window_bbox in windows
    ]

    for window, prediction in zip(windows, model.predict(np.array(window_imgs))):
        # print(prediction)
        if prediction:
            yield window


def get_bboxes_from_heatmap(heatmap):
    label_map, count = sk_label(heatmap)

    def get_bbox(num):
        nonzeroy, nonzerox = (label_map == num).nonzero()
        return np.min(nonzerox), np.min(nonzeroy), np.max(nonzerox), np.max(nonzeroy)

    return [get_bbox(num + 1) for num in range(count)]


def main(img, show=False):
    heatmap = np.zeros(img.shape[:2], dtype=int)
    # bboxes = []
    for positive_window in slide_window_predict(img, batch_size=64, xy_overlap=[0.8, 0.8], xy_window=[64, 64], y_start_stop=[400, 656]):  # , xy_window=[64, 64], xy_overlap=[0.25, 0.25]):
        # print(positive_window)
        heatmap = add_heat(heatmap, positive_window)
        # bboxes.append(positive_window)
    heatmap = apply_threshold(heatmap, 2)
    bboxes = get_bboxes_from_heatmap(heatmap)
    # bboxes=slide_window(img.shape)
    draw_img = draw_bboxes(img, bboxes, show=show)
    return draw_img


def main_find_cars(img_or_path, show=False):
    img, draw_img = imread(img_or_path, draw=True)
    heatmap = np.zeros(img.shape[:2], dtype=int)
    bboxes = []
    for window in [64]:
        for positive_window in model.find_cars(img,
                                               y_start_stop=[400, 656],
                                               window=window
                                               ):
            # print(positive_window)
            heatmap = add_heat(heatmap, positive_window)
            # bboxes.append(positive_window)
    heatmap = apply_threshold(heatmap, 3)
    bboxes = get_bboxes_from_heatmap(heatmap)

    draw_img = draw_bboxes(draw_img, bboxes, show=show)
    return draw_img


class FrameQueue:

    def __init__(self, capacity):
        self.capacity = capacity
        self.frames = []

    def enqueue(self, item):
        self.frames.append(item)
        if len(self.frames) > self.capacity:
            self.frames.pop(0)

    def sum_frames(self):
        return np.sum(self.frames, axis=0)


class Detector:

    def __init__(self, save=False):
        self.queue = FrameQueue(25)
        self.save = save
        self.seq = 0

    def detect(self, img):
        self.seq += 1
        img, draw_img = imread(img, draw=True)
        heatmap = np.zeros(img.shape[:2], dtype=int)
        # bboxes = []
        for window in [64]:
            for positive_window in model.find_cars(img,
                                                   y_start_stop=[400, 656],
                                                   window=window
                                                   ):
                # print(positive_window)
                heatmap = add_heat(heatmap, positive_window)
            # bboxes.append(positive_window)

        self.queue.enqueue(heatmap)
        heatmap = self.queue.sum_frames()

        heatmap = apply_threshold(heatmap, 20)
        bboxes = get_bboxes_from_heatmap(heatmap)
        draw_img = draw_bboxes(draw_img, bboxes, show=False)
        if self.save:
            imwrite(draw_img, "%s_%d.png" % (self.save, self.seq), source_color="RGB")
        return draw_img


def apply_video(input_video_path, output_path, save=False):
    from moviepy.editor import VideoFileClip
    clip = VideoFileClip(input_video_path).fl_image(Detector(save=save).detect)
    clip.write_videofile(output_path, audio=False, threads=8)
