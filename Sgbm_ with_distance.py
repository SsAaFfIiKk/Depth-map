import cv2
import h5py
import os
import numpy as np


with h5py.File("camera_maps.hdf5", "r") as f:
    LeftX = f["LeftX"][()]
    LeftY = f["LeftY"][()]
    RightX = f["RightX"][()]
    RightY = f["RightY"][()]

left = cv2.VideoCapture(4)
right = cv2.VideoCapture(2)


class DepthMapCreator:
    def __init__(self, LeftX, LeftY, RightX, RightY):
        self.LeftX = LeftX
        self.LeftY = LeftY
        self.RightX = RightX
        self.RightY = RightY
        self.kernel= np.ones((3,3),np.uint8)
        self.mode = cv2.STEREO_SGBM_MODE_SGBM_3WAY

    def set_default_settings(self, min_disparity=2, num_disparities=128, uniqueness_ratio=10,
                             speckle_window_size=100, speckle_range=32, disp12_max_diff=5, P1=216, P2=864,
                             block_size=3, pre_filter_cap=63):
        self.min_disparity = min_disparity
        self.num_disparities = num_disparities  # должен быть кратен 16
        self.uniqueness_ratio = uniqueness_ratio  # запас в процентах, обычно 5 - 15
        self.speckle_window_size = speckle_window_size  # максимальный размер сглаженных областей диспаратности, обычно 50 - 200
        self.speckle_range = speckle_range  # макс изменение несоответствия в каждом подключенном компоненте, обычно 1 - 2
        self.disp12_max_diff = disp12_max_diff  # макс допустиамая разница, если отрицательна то отключается
        self.P1 = P1  # гладкость диспарсии
        self.P2 = P2  # должен быть больше P1
        self.block_size = block_size
        self.pre_filter_cap = pre_filter_cap


    def load_map_settings(self):
        with h5py.File("map_settings.hdf5", "r") as m:
            self.min_disparity = m["min"][()]
            self.num_disparities = m["num"][()]
            self.uniqueness_ratio = m["uniq"][()]
            self.speckle_window_size = m["speckw"][()]
            self.speckle_range = m["speckr"][()]
            self.disp12_max_diff = m["disp"][()]
            self.P1 = m["P1"][()]
            self.P2 = m["P2"][()]
            self.block_size = m["block"][()]
            self.pre_filter_cap = m["pre"][()]

    def create_stereoSGBM(self):
        self.stereoSGBM = cv2.StereoSGBM_create(minDisparity=self.min_disparity,
                                                numDisparities=self.num_disparities,
                                                uniquenessRatio=self.uniqueness_ratio,
                                                speckleWindowSize=self.speckle_window_size,
                                                speckleRange=self.speckle_range,
                                                disp12MaxDiff=self.disp12_max_diff,
                                                P1=self.P1,
                                                P2=self.P2,
                                                blockSize=self.block_size,
                                                preFilterCap=self.pre_filter_cap,
                                                mode = self.mode)

    def remap(self, img_l, img_r):
        # remaped_l = cv2.remap(img_l, self.LeftX, self.LeftY, cv2.INTER_LINEAR)
        # remaped_r = cv2.remap(img_r, self.RightX, self.RightY, cv2.INTER_LINEAR)
        # remaped_l = cv2.remap(img_l, self.LeftX, self.LeftY, cv2.INTER_LINEAR, cv2.BORDER_CONSTANT)
        # remaped_r = cv2.remap(img_r, self.RightX, self.RightY, cv2.INTER_LINEAR, cv2.BORDER_CONSTANT)
        remaped_l = cv2.remap(img_l, self.LeftX, self.LeftY, cv2.INTER_LANCZOS4, cv2.BORDER_CONSTANT)
        remaped_r = cv2.remap(img_r, self.RightX, self.RightY, cv2.INTER_LANCZOS4, cv2.BORDER_CONSTANT)
        return remaped_l, remaped_r

    def img_pair_to_gray(self, left_img, right_img):
        gray_img_L = cv2.cvtColor(left_img, cv2.COLOR_BGR2GRAY)
        gray_img_R = cv2.cvtColor(right_img, cv2.COLOR_BGR2GRAY)
        return gray_img_L, gray_img_R

    def create_depth_map(self, gray_l, gray_r):
        self.disparity = self.stereoSGBM.compute(gray_l, gray_r)
        map = cv2.normalize(self.disparity, self.disparity, 0, 255, cv2.NORM_MINMAX)
        map = np.uint8(map)
        return map

    def get_distance(self, event, x, y, glags, params):
        if event == cv2.EVENT_LBUTTONDBLCLK:
            zone = self.disparity[y-1:y+1, x-1:x+1]
            average = np.mean(zone)
            distance = -593.97 * average ** 3 + 1506.8 * average ** 2 - 1373.1 * average + 522.06
            distance = np.around(distance * 0.01, decimals=2)
            print("Distance: " + str(distance) + " m")




dm_computer = DepthMapCreator(LeftX, LeftY, RightX, RightY)
if os.path.exists("map_settings.hdf5"):
    dm_computer.load_map_settings()
else:
    dm_computer.set_default_settings()
dm_computer.create_stereoSGBM()

while True:
    r_ret, r_frame = right.read()
    l_ret, l_frame = left.read()

    remap_l, remap_r = dm_computer.remap(l_frame, r_frame)
    gray_l, gray_r = dm_computer.img_pair_to_gray(remap_l, remap_r)

    depth_map = dm_computer.create_depth_map(gray_l, gray_r)

    cv2.imshow("Depth map", depth_map)
    cv2.setMouseCallback("Depth map", dm_computer.get_distance)

    if cv2.waitKey(1) & 0xFF == 27:
        break

right.release()
left.release()
cv2.destroyAllWindows()
