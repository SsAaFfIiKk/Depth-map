import cv2
import os
import numpy as np
from time import time


def save_images(r_frame, l_frame, f_number):
    str_f_number = str(f_number)
    path_to_save = os.path.join(folder_sav, str_f_number)
    cv2.imwrite(path_to_save + "_veb_r.png", r_frame)
    cv2.imwrite(path_to_save + "_veb_l.png", l_frame)
    print(str_f_number + " is save")
    if str_f_number == 29:
        print("save complete")


criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

left = cv2.VideoCapture(4)
right = cv2.VideoCapture(2)

frame_number = 0
save = False
folder_sav = ""
direct = os.listdir()


if "set_0" not in direct:
    os.mkdir("set_0")
    folder_sav = "set_0"
else:
    for i in direct:
        if len(i) >= 5:
            if i[:4] == "set_":
                x = i[4:]
                y = int(x) + 1
                folder_sav = i[:4] + str(y)
            if folder_sav in direct:
                x = folder_sav[4:]
                y = int(x) + 1
                folder_sav = folder_sav[:4] + str(y)
    os.mkdir(folder_sav)

st_time = time()
while True:
    r_ret, r_frame = right.read()
    l_ret, l_frame = left.read()

    R = np.copy(r_frame)
    L = np.copy(l_frame)

    gray_r = cv2.cvtColor(r_frame, cv2.COLOR_BGR2GRAY)
    gray_l = cv2.cvtColor(l_frame, cv2.COLOR_BGR2GRAY)

    ret_r, corners_r = cv2.findChessboardCorners(gray_r, (6, 9), None)
    ret_l, corners_l = cv2.findChessboardCorners(gray_l, (6, 9), None)

    if ret_r and ret_l:

        cor_r = cv2.cornerSubPix(gray_r, corners_r, (11, 11), (-1, -1), criteria)
        chest_r = cv2.drawChessboardCorners(R, (6, 9), cor_r, ret_r)

        cor_l = cv2.cornerSubPix(gray_l, corners_l, (11, 11), (-1, -1), criteria)
        chest_l = cv2.drawChessboardCorners(L, (6, 9), cor_l, ret_l)

    cv2.imshow("R l", np.hstack([R, L]))
    end_time = time()

    if frame_number < 30 and ret_r and ret_l:
        save = True
    elif frame_number > 29:
        save = False
    else:
        st_time = time()

    if save:
        if end_time - st_time > 5:
            save_images(r_frame, l_frame, frame_number)
            frame_number += 1
            st_time = time()

    if cv2.waitKey(1) & 0xFF == 27:
        break

right.release()
left.release()
cv2.destroyAllWindows()
