import cv2
import h5py
import numpy as np
from random import randint

rows = 6
cols = 9
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

with h5py.File("stereo_maps.hdf5", "r") as f:
    LeftX = f["LeftX"]
    LeftY = f["LeftY"]
    RightX = f["RightX"]
    RightY = f["RightY"]

corners_r = []
corners_l = []
w = None


def load_picture():
    img_r = cv2.imread("0_r.png")
    img_l = cv2.imread("0_l.png")
    return img_r, img_l

def remap(img_r, img_l,  RightX, RightY, LeftX, LeftY):
    remaped_r = cv2.remap(img_r, RightX, RightY, cv2.INTER_LINEAR)
    remaped_l = cv2.remap(img_l, LeftX, LeftY, cv2.INTER_LINEAR)
    return remaped_r, remaped_l

def find_cornes(gray_r, gray_l, rows_cols, criteria):
    ret_r, corners_r = cv2.findChessboardCorners(gray_r, rows_cols, None)
    ret_l, corners_l = cv2.findChessboardCorners(gray_l, rows_cols, None)

    if ret_r and ret_l:
        corners_r = cv2.cornerSubPix(gray_r, corners_r, (11, 11), (-1, -1), criteria)
        corners_l = cv2.cornerSubPix(gray_l, corners_l, (11, 11), (-1, -1), criteria)
    return corners_r, corners_l


def draw_eplines(corners_r, corners_l, stacked_img):
    w = stacked_img.shape[1]//2
    for l_point, r_point in zip(corners_l, corners_r):
        color = (randint(0, 255), randint(0, 255), randint(0, 255))
        lp, rp = l_point[0], r_point[0]
        xl, yl = tuple(int(val) for val in lp)
        xyr = tuple(int(val) for val in rp)
        xl += w
        cv2.line(stacked_img, xyr, (xl, yl), color, 1)
        cv2.circle(stacked_img, xyr, 2, color, 2)
        cv2.circle(stacked_img, (xl, yl), 2, color, 2)
    return stacked_img

img_r, img_l = load_picture()

grey_r = cv2.cvtColor(img_r, cv2.COLOR_BGR2GRAY)
grey_l = cv2.cvtColor(img_l, cv2.COLOR_BGR2GRAY)

corners_r, corners_l = find_cornes(grey_r, grey_l, (rows, cols), criteria)
stacked = np.hstack([img_r, img_l])

stacked = draw_eplines(corners_r, corners_l, stacked)

cv2.imshow("ep_lines", stacked)
cv2.waitKey()
cv2.destroyAllWindows()
