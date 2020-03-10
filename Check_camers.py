import cv2
import h5py
import numpy as np


def nothing(x):
    pass


# with h5py.File("camera_maps.hdf5", "r") as f:
#     LeftX = f["LeftX"][()]
#     LeftY = f["LeftY"][()]
#     RightX = f["RightX"][()]
#     RightY = f["RightY"][()]

left = cv2.VideoCapture(4)
right = cv2.VideoCapture(2)

trackbar_window_name = "off/on red lines"
cv2.namedWindow(trackbar_window_name, cv2.WINDOW_NORMAL)
cv2.createTrackbar("off/on", trackbar_window_name, 0, 1, nothing)

while True:

    s = cv2.getTrackbarPos("off/on", trackbar_window_name)

    _, frame_l = left.read()
    _, frame_r = right.read()

    # remaped_l = cv2.remap(frame_l, LeftX, LeftY, cv2.INTER_LANCZOS4, cv2.BORDER_CONSTANT)
    # remaped_r = cv2.remap(frame_r, RightX, RightY, cv2.INTER_LANCZOS4, cv2.BORDER_CONSTANT)

    if s == 1:
        for line in range(0, int(frame_l.shape[0]/20)):
            frame_l[line*20, :] = (0, 0, 255)
            frame_r[line * 20, :] = (0, 0, 255)

            # remaped_l[line * 20, :] = (0, 0, 255)
            # remaped_r[line * 20, :] = (0, 0, 255)

    cv2.imshow("R L", np.hstack([frame_r, frame_l]))
    # cv2.imshow("rR rL", np.hstack([remaped_r, remaped_l]))

    if cv2.waitKey(1) & 0xFF == 27:
        break

cv2.destroyAllWindows()
