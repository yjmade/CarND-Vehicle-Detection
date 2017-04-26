# -*- coding: utf-8 -*-

import os
import sys
sys.path.insert(1, os.path.dirname(__file__))

import cv2
import glob
import pickle
import numpy as np
from sklearn.model_selection import train_test_split

from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from joblib import Parallel, delayed

import features
from utils import log_time  # , print_loop


# TODO: Tweak these parameters and see how the results change.
colorspace = 'YCrCb'  # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
orient = 9
pix_per_cell = 8
cell_per_block = 2
hog_channel = "ALL"  # Can be 0, 1, 2, or "ALL"
spatial_size = (32, 32)
hist_bins = 32
scale = 1.5


MODEL_PATH = os.path.join(os.path.dirname(__file__), "svc.pickle")
DATA_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")


def imread(path_or_img, source_color="RGB", draw=False):
    if isinstance(path_or_img, str):
        img = cv2.imread(path_or_img)
        source_color = "BGR"
    else:
        img = path_or_img

    conv_img = features.color_cov(img, colorspace, source_color=source_color)
    conv_img = conv_img.astype(np.float32) / 255

    if draw:
        return conv_img, features.color_cov(img, source_color, "RGB")
    return conv_img


def imwrite(img, output_path, source_color="RGB"):
    img = features.color_cov(img, "BGR", source_color)
    cv2.imwrite(output_path, img)


def extract_feature(img):
    hog_features = features.hog_feature(img, orient=orient, pix_per_cell=pix_per_cell, cell_per_block=cell_per_block, hog_channel=hog_channel)
    spatial_features = features.bin_spatial(img, size=spatial_size)
    hist_features = features.color_hist(img, nbins=hist_bins)
    return np.hstack((spatial_features, hist_features, hog_features))


def extract_feature_from_path(path):
    return extract_feature(imread(path))


def train_model():
    # Divide up into cars and notcars
    # images = glob.iglob('*.jpeg', recursive=True)
    cars = glob.iglob(os.path.join(DATA_PATH, "vehicles", "**", "*.png"), recursive=True)
    notcars = glob.iglob(os.path.join(DATA_PATH, "non-vehicles", "**", "*.png"), recursive=True)
    # for image in images:
    #     if 'image' in image or 'extra' in image:
    #         notcars.append(image)
    #     else:
    #         cars.append(image)

    with log_time("extract car features"):
        car_features = Parallel(n_jobs=-1, max_nbytes=None, verbose=5)(
            delayed(extract_feature_from_path)(img_path) for img_path in cars
        )
    with log_time("extract noncar features"):
        notcar_features = Parallel(n_jobs=-1, max_nbytes=None, verbose=5)(
            delayed(extract_feature_from_path)(img_path) for img_path in notcars
        )

    # Create an array stack of feature vectors
    X = np.vstack((car_features, notcar_features)).astype(np.float64)
    # Fit a per-column scaler
    X_scaler = StandardScaler().fit(X)
    # Apply the scaler to X
    scaled_X = X_scaler.transform(X)

    # Define the labels vector
    y = np.hstack((np.ones(len(car_features)), np.zeros(len(notcar_features))))

    # Split up data into randomized training and test sets
    rand_state = np.random.randint(0, 100)
    X_train, X_test, y_train, y_test = train_test_split(scaled_X, y, test_size=0.2, random_state=rand_state)

    print('Using:', orient, 'orientations', pix_per_cell,
          'pixels per cell and', cell_per_block, 'cells per block')
    print('Feature vector length:', len(X_train[0]))
    # Use a linear SVC
    svc = LinearSVC()
    # Check the training time for the SVC
    with log_time("train"):
        svc.fit(X_train, y_train)
    # Check the score of the SVC
    print('Test Accuracy of SVC = ', round(svc.score(X_test, y_test), 4))
    # Check the prediction time for a single sample
    n_predict = 10
    with log_time("evaluate", n_predict, "items"):
        predicts = svc.predict(X_test[0:n_predict])
    print('My SVC predicts: ', predicts)
    print('For these', n_predict, 'labels: ', y_test[0:n_predict])

    with open(MODEL_PATH, "wb") as f:
        pickle.dump([svc, X_scaler], f)


class Model:

    def __init__(self, clf, X_scaler):   # noqa
        self.clf = clf
        self.X_scaler = X_scaler

    def predict(self, images):
        features = self.X_scaler.transform([
            extract_feature(img)
            for img in images
        ])
        return self.clf.predict(features)

    def find_cars(
        self,
        img,
        y_start_stop,
        window=64,
        cells_per_step=2  # Instead of overlap, define how many cells to step
    ):

        ystart, ystop = y_start_stop
        ystart = ystart or 0
        ystop = ystop or img.shape[0]

        ctrans_tosearch = img[ystart:ystop, :, :]
        if scale != 1:
            imshape = ctrans_tosearch.shape
            ctrans_tosearch = cv2.resize(ctrans_tosearch, (np.int(imshape[1] / scale), np.int(imshape[0] / scale)))

        ch1 = ctrans_tosearch[:, :, 0]
        ch2 = ctrans_tosearch[:, :, 1]
        ch3 = ctrans_tosearch[:, :, 2]

        # Define blocks and steps as above
        nxblocks = (ch1.shape[1] // pix_per_cell) - 1
        nyblocks = (ch1.shape[0] // pix_per_cell) - 1
        # nfeat_per_block = orient * cell_per_block**2
        # 64 was the orginal sampling rate, with 8 cells and 8 pix per cell

        nblocks_per_window = (window // pix_per_cell) - 1
        nxsteps = (nxblocks - nblocks_per_window) // cells_per_step
        nysteps = (nyblocks - nblocks_per_window) // cells_per_step

        # Compute individual channel HOG features for the entire image

        hog1 = features.get_hog_features(ch1, orient, pix_per_cell, cell_per_block, feature_vec=False)
        hog2 = features.get_hog_features(ch2, orient, pix_per_cell, cell_per_block, feature_vec=False)
        hog3 = features.get_hog_features(ch3, orient, pix_per_cell, cell_per_block, feature_vec=False)

        for xb in range(nxsteps):
            for yb in range(nysteps):
                ypos = yb * cells_per_step
                xpos = xb * cells_per_step
                # Extract HOG for this patch
                hog_feat1 = hog1[ypos:ypos + nblocks_per_window, xpos:xpos + nblocks_per_window].ravel()
                hog_feat2 = hog2[ypos:ypos + nblocks_per_window, xpos:xpos + nblocks_per_window].ravel()
                hog_feat3 = hog3[ypos:ypos + nblocks_per_window, xpos:xpos + nblocks_per_window].ravel()
                hog_features = np.hstack((hog_feat1, hog_feat2, hog_feat3))

                xleft = xpos * pix_per_cell
                ytop = ypos * pix_per_cell

                # Extract the image patch
                subimg = cv2.resize(ctrans_tosearch[ytop:ytop + window, xleft:xleft + window], (64, 64))

                # Get color features
                spatial_features = features.bin_spatial(subimg, size=spatial_size)
                hist_features = features.color_hist(subimg, nbins=hist_bins)

                # Scale features and make a prediction
                test_features = self.X_scaler.transform(np.hstack((spatial_features, hist_features, hog_features)).reshape(1, -1))
                test_prediction = self.clf.predict(test_features)

                if test_prediction == 1:
                    xbox_left = np.int(xleft * scale)
                    ytop_draw = np.int(ytop * scale)
                    win_draw = np.int(window * scale)
                    yield xbox_left, ytop_draw + ystart, xbox_left + win_draw, ytop_draw + win_draw + ystart


def get_model():
    with open(MODEL_PATH, "rb") as f:
        clf, X_scaler = pickle.load(f)

    return Model(clf, X_scaler)
