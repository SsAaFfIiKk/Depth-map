import os
import cv2
import h5py
import numpy as np


objpoints = []
imgpoints_r = []
imgpoints_l = []

r_imgs = []
l_imgs = []

imgs_folder = "set_0"
img_num = 0
img_size = None
photos_lst = os.listdir(imgs_folder)

rows = 6
cols = 9

objp = np.zeros((rows * cols, 3), np.float32)
objp[:, :2] = np.mgrid[0:rows, 0:cols].T.reshape(-1, 2)
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
stereocalib_criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

stereocalib_flags = 0
stereocalib_flags |= cv2.CALIB_FIX_INTRINSIC
# stereocalib_flags |= cv2.CALIB_USE_INTRINSIC_GUESS
# stereocalib_flags |= cv2.CALIB_FIX_PRINCIPAL_POINT
stereocalib_flags |= cv2.CALIB_FIX_FOCAL_LENGTH
# stereocalib_flags |= cv2.CALIB_FIX_ASPECT_RATIO
# stereocalib_flags |= cv2.CALIB_SAME_FOCAL_LENGTH
stereocalib_flags |= cv2.CALIB_ZERO_TANGENT_DIST
# stereocalib_flags |= cv2.CALIB_RATIONAL_MOD EL
# stereocalib_flags |= cv2.CALIB_FIX_K1
# stereocalib_flags |= cv2.CALIB_FIX_K2
# stereocalib_flags |= cv2.CALIB_FIX_K3
# stereocalib_flags |= cv2.CALIB_FIX_K4
# stereocalib_flags |= cv2.CALIB_FIX_K5
# stereocalib_flags |= cv2.CALIB_FIX_K6

left = cv2.VideoCapture(4)
right = cv2.VideoCapture(2)


def nothing(x):
    pass


print("Ищем доску")
while len(r_imgs) + len(l_imgs) < len(photos_lst):
    r_img_name = str(img_num) + "_veb_r.png"
    l_img_name = str(img_num) + "_veb_l.png"

    img_r = cv2.imread(os.path.join(imgs_folder, r_img_name))
    img_l = cv2.imread(os.path.join(imgs_folder, l_img_name))
    r_imgs.append(img_r)
    l_imgs.append(img_l)

    gray_r = cv2.cvtColor(img_r, cv2.COLOR_BGR2GRAY)
    gray_l = cv2.cvtColor(img_l, cv2.COLOR_BGR2GRAY)

    ret_r, corners_r = cv2.findChessboardCorners(gray_r, (rows, cols))
    ret_l, corners_l = cv2.findChessboardCorners(gray_l, (rows, cols))

    if ret_r and ret_l:
        corners_r = cv2.cornerSubPix(gray_r, corners_r, (11, 11), (-1, -1), criteria)
        imgpoints_r.append(corners_r)
        cv2.drawChessboardCorners(img_r, (rows, cols), corners_r, ret_r)

        corners_l = cv2.cornerSubPix(gray_l, corners_l, (11, 11), (-1, -1), criteria)
        imgpoints_l.append(corners_l)
        cv2.drawChessboardCorners(img_l, (rows, cols), corners_l, ret_l)

        objpoints.append(objp)


# показывает нахождение доски
    cv2.imshow(str(img_num), np.hstack((img_r, img_l)))
    cv2.waitKey(420)
    cv2.destroyAllWindows()

    img_num += 1

img_size = gray_l.shape[::-1]

print("Калибруемся по отделности")
# калибровка камер по отдельности
_, mtx_r, dist_r, _, _ = cv2.calibrateCamera(objpoints, imgpoints_r, img_size, None, None)
_, mtx_l, dist_l, _, _ = cv2.calibrateCamera(objpoints, imgpoints_l, img_size, None, None)

# применеие новых параметров чтобы исправить изображения
r_img = cv2.imread(os.path.join(imgs_folder, r_img_name))
undistortedImg_r = cv2.undistort(r_img, mtx_r, dist_r)

l_img = cv2.imread(os.path.join(imgs_folder, l_img_name))
undistortedImg_l = cv2.undistort(l_img, mtx_l, dist_l)

cv2.imshow(str(29) + "_veb_l", undistortedImg_l)
cv2.imshow(str(29) + "_veb_r", undistortedImg_r)
cv2.waitKey()
cv2.destroyAllWindows()

print("Калибруемся вместе")
_, mtx_L, dist_L, mtx_R, dist_R, R, T, _, _ = cv2.stereoCalibrate(objpoints, imgpoints_l, imgpoints_r,
                                                                  mtx_l, dist_l,
                                                                  mtx_r, dist_r,
                                                                  img_size,
                                                                  flags=stereocalib_flags,
                                                                  criteria=stereocalib_criteria)

R_l, R_r, P_l, P_r, Q, roi_l, roi_r = cv2.stereoRectify(mtx_L, dist_L,
                                                        mtx_R, dist_R,
                                                        img_size,
                                                        R, T, alpha=0, flags=cv2.CALIB_ZERO_DISPARITY)
print(mtx_L)
print(mtx_R)


RightX, RightY = cv2.initUndistortRectifyMap(mtx_r, dist_r, R_r, P_r, img_size, cv2.CV_32FC1)
LeftX, LeftY = cv2.initUndistortRectifyMap(mtx_l, dist_l, R_l, P_l, img_size, cv2.CV_32FC1)

trackbar_window_name = "off/on red lines"
cv2.namedWindow(trackbar_window_name, cv2.WINDOW_NORMAL)
cv2.createTrackbar("off/on", trackbar_window_name, 0, 1, nothing)
img_num = 0

while True:
    key = cv2.waitKey(1) & 0xFF
    s = cv2.getTrackbarPos("off/on", trackbar_window_name)

    ret_r, frame_r = right.read()
    ret_l, frame_l = left.read()

    remaped_r = cv2.remap(frame_r, RightX, RightY, cv2.INTER_LINEAR)
    remaped_l = cv2.remap(frame_l, LeftX, LeftY, cv2.INTER_LINEAR)

    # Draw Red lines
    if s == 1:
        for line in range(0, int(remaped_l.shape[0]/20)):
            remaped_l[line*20, :] = (0, 0, 255)
            remaped_r[line*20, :] = (0, 0, 255)

    cv2.imshow("R L", np.hstack([remaped_r, remaped_l]))
    # cv2.imshow("R", remaped_r)
    # cv2.imshow("L", remaped_l)

    if key == 27:
        break

    elif key == ord("s"):
        with h5py.File("camera_maps.hdf5", "w") as f:
            f.create_dataset("RightX", data=RightX)
            f.create_dataset("RightY", data=RightY)
            f.create_dataset("LeftX", data=LeftX)
            f.create_dataset("LeftY", data=LeftY)
            f.create_dataset("Q", data=Q)
            f.create_dataset("roi_r", data=roi_r)
            f.create_dataset("roi_l", data=roi_l)
        break

right.release()
left.release()
cv2.destroyAllWindows()
