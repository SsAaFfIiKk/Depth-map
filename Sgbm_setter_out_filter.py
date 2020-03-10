import cv2
import xlwt
import h5py
import numpy as np

with h5py.File("camera_maps.hdf5", "r") as f:
    LeftX = f["LeftX"][()]
    LeftY = f["LeftY"][()]
    RightX = f["RightX"][()]
    RightY = f["RightY"][()]

left = cv2.VideoCapture(4)
right = cv2.VideoCapture(2)

trackbar_window_name = "Map settings"
cv2.namedWindow(trackbar_window_name, cv2.WINDOW_NORMAL)


class DepthMapCreator:
    def __init__(self, LeftX, LeftY, RightX, RightY):
        self.LeftX = LeftX
        self.LeftY = LeftY
        self.RightX = RightX
        self.RightY = RightY
        self.zone = 0
        self.distance = 0
        self.list_disaprity = []
        self.list_distance = []
        self.kernel = np.ones((3, 3), np.uint8)
        self.stereoSGBM = None
        self.is_changes = False
        self.mode = cv2.STEREO_SGBM_MODE_SGBM_3WAY

    def reset_changes(self):
        self.is_changes = False

    # настройки карты по умолчанию
    def set_default_settings(self, min_disparity=2, num_disparities=128, uniqueness_ratio=10,
                             speckle_window_size=100, speckle_range=32, disp12_max_diff=5, P1=216, P2=864, block_size=3,
                             pre_filter_cap=63):
        self.min_disparity = min_disparity
        self.num_disparities = num_disparities
        self.uniqueness_ratio = uniqueness_ratio
        self.speckle_window_size = speckle_window_size
        self.speckle_range = speckle_range
        self.disp12_max_diff = disp12_max_diff
        self.P1 = P1
        self.P2 = P2
        self.block_size = block_size
        self.pre_filter_cap = pre_filter_cap

    # создание бегунком для настройки карты
    def create_trackbar(self, window_name):
        cv2.createTrackbar("minDisparity", window_name, 2, 360, self.callback_minD)
        cv2.createTrackbar("numDisparities", window_name, self.num_disparities//16, 20, self.callback_numD)
        cv2.createTrackbar("uniquenessRatio", window_name, self.uniqueness_ratio, 15, self.callback_uniqR)
        cv2.createTrackbar("speckleWindowSize ", window_name, self.speckle_window_size, 200, self.callback_speckW)
        cv2.createTrackbar("speckleRange ", window_name, self.speckle_range, 40, self.callback_speckR)
        cv2.createTrackbar("disp12MaxDiff", window_name, self.disp12_max_diff, 20, self.callback_disp)
        cv2.createTrackbar("P1", window_name, self.P1, 600, self.callback_P1)
        cv2.createTrackbar("P2", window_name, self.P2, 2400, self.callback_P2)
        cv2.createTrackbar("blockSize", window_name, self.block_size, 11, self.callback_block)
        cv2.createTrackbar("preFilterCap", window_name, self.pre_filter_cap, 160, self.callback_preF)

    def callback_minD(self, x):
        self.is_changes = True
        self.min_disparity = cv2.getTrackbarPos("minDisparity", trackbar_window_name)
        if self.min_disparity < 180:
            self.min_disparity *= -1
        else:
            self.min_disparity -= 180

    def callback_numD(self, x):
        self.is_changes = True
        self.num_disparities = cv2.getTrackbarPos("numDisparities", trackbar_window_name) * 16
        if self.num_disparities < 1:
            self.num_disparities = 16

    def callback_uniqR(self, x):
        self.is_changes = True
        self.uniqueness_ratio = cv2.getTrackbarPos("uniquenessRatio", trackbar_window_name)

    def callback_speckW(self, x):
        self.is_changes = True
        self.speckle_window_size = cv2.getTrackbarPos("speckleWindowSize ", trackbar_window_name)

    def callback_speckR(self, x):
        self.is_changes = True
        self.speckle_range = cv2.getTrackbarPos("speckleRange ", trackbar_window_name)

    def callback_disp(self,x):
        self.is_changes = True
        self.disp12_max_diff = cv2.getTrackbarPos("disp12MaxDiff", trackbar_window_name)

    def callback_P1(self,x):
        self.is_changes = True
        self.P1 = cv2.getTrackbarPos("P1", trackbar_window_name)

    def callback_P2(self,x):
        self.is_changes = True
        self.P2 = cv2.getTrackbarPos("P2", trackbar_window_name)
        if self.P2 < self.P1:
            self.P2 = self.P1 + 1

    def callback_block(self,x):
        self.is_changes = True
        self.block_size = cv2.getTrackbarPos("blockSize", trackbar_window_name)

    def callback_preF(self,x):
        self.is_changes = True
        self.pre_filter_cap = cv2.getTrackbarPos("preFilterCap", trackbar_window_name)
        if self.pre_filter_cap < 1:
            self.pre_filter_cap = 1

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

        # self.stereoSGBM = cv2.StereoSGBM_create(minDisparity=self.min_disparity,
        #                                         numDisparities=self.num_disparities,
        #                                         mode=self.mode)


    def remap(self, img_l, img_r):
        # remaped_l = cv2.remap(img_l, self.LeftX, self.LeftY, cv2.INTER_LINEAR)
        # remaped_r = cv2.remap(img_r, self.RightX, self.RightY, cv2.INTER_LINEAR)
        # remaped_l = cv2.remap(img_l, self.LeftX, self.LeftY, cv2.INTER_LINEAR, cv2.BORDER_CONSTANT)
        # remaped_r = cv2.remap(img_r, self.RightX, self.RightY, cv2.INTER_LINEAR, cv2.BORDER_CONSTANT)
        remaped_l = cv2.remap(img_l, self.LeftX, self.LeftY, cv2.INTER_LANCZOS4, cv2.BORDER_CONSTANT)
        remaped_r = cv2.remap(img_r, self.RightX, self.RightY, cv2.INTER_LANCZOS4, cv2.BORDER_CONSTANT)
        return remaped_l, remaped_r

    def remaped_to_grey(self, remaped_l, remaped_r):
        gray_l = cv2.cvtColor(remaped_l, cv2.COLOR_BGR2GRAY)
        gray_r = cv2.cvtColor(remaped_r, cv2.COLOR_BGR2GRAY)
        return gray_l, gray_r

    def create_depth_map(self, gray_l, gray_r):
        self.disparity = self.stereoSGBM.compute(gray_l, gray_r)
        # cv2.filterSpeckles(self.disparity, 0, 64, 32)
        # self.disparity = cv2.morphologyEx(self.disparity, cv2.MORPH_CLOSE, self.kernel)
        map = cv2.normalize(self.disparity, self.disparity, 0, 255, cv2.NORM_MINMAX)
        map = np.uint8(map)
        return map

    def start_collection_data(self, event, x, y, flags, params):
        if event == cv2.EVENT_LBUTTONDBLCLK:
            self.get_disparity(x, y)
            self.input_distance()
            self.append()

    def get_disparity(self, x, y):
        self.zone = self.disparity[y - 1:y + 1, x - 1:x + 1]
        self.zone = np.mean(self.zone)*0.01

    def input_distance(self):
        if self.distance == 0:
            try:
                self.interval = int(input("Введите на сколько будет сдивигаться обьект: "))
                self.distance = int(input("Введите начальную дистанцию до обьекта: "))
            except:
                print("Введен не верный формат")
        else:
            self.distance += self.interval
            print("последняя дистанция: ", self.distance)

    def append(self):
        self.list_disaprity.append(self.zone)
        self.list_distance.append(self.distance)

    def remove_last(self):
        del self.list_distance[-1]
        del self.list_disaprity[-1]
        self.distance -= self.interval
        print(self.distance)

    def save_in_excel(self):
        if len(self.list_distance) and len(self.list_disaprity) > 1:
            book = xlwt.Workbook(encoding="utf-8")
            sheet1 = book.add_sheet("Лист 1")

            sheet1.write(0, 0, "Distance")
            sheet1.write(0, 1, "Disparity")

            i = 1
            for n, m in zip(self.list_disaprity, self.list_distance):
                sheet1.write(i, 1, n)
                sheet1.write(i, 0, m)
                i += 1

            book.save("Distance and disparity.xls")
            print("Save complete")

    def save_map_settings(self):
        with h5py.File("map_settings_out_filter.hdf5", "w") as f:
            f.create_dataset("min",data=self.min_disparity)
            f.create_dataset("num",data=self.num_disparities)
            f.create_dataset("uniq",data=self.uniqueness_ratio)
            f.create_dataset("speckw",data=self.speckle_window_size)
            f.create_dataset("speckr",data=self.speckle_range)
            f.create_dataset("disp",data=self.disp12_max_diff)
            f.create_dataset("P1",data=self.P1)
            f.create_dataset("P2",data=self.P2)
            f.create_dataset("block",data=self.block_size)
            f.create_dataset("pre",data=self.pre_filter_cap)


dm_computer = DepthMapCreator(LeftX, LeftY, RightX, RightY)
dm_computer.set_default_settings()
dm_computer.create_stereoSGBM()
dm_computer.create_trackbar(trackbar_window_name)

while True:
    if dm_computer.is_changes:
        dm_computer.create_stereoSGBM()
        dm_computer.reset_changes()

    _, r_frame = right.read()
    _, l_frame = left.read()

    remaped_l, remaped_r = dm_computer.remap(l_frame, r_frame)
    gray_l, gray_r = dm_computer.remaped_to_grey(remaped_l, remaped_r)

    depth_map = dm_computer.create_depth_map(gray_l, gray_r)

    cv2.imshow("Depth Map", depth_map)
    cv2.setMouseCallback("Depth Map", dm_computer.start_collection_data)

    if cv2.waitKey(1) & 0xFF == 27:
        dm_computer.save_in_excel()
        # dm_computer.save_map_settings()
        break

    if cv2.waitKey(1) & 0xFF == 32:
        dm_computer.remove_last()
        print("Последний ввод удален")

    if cv2.waitKey(1) & 0xFf == ord("s"):
        cv2.imwrite("Depth_out_filter.png", depth_map)
        print("Depth map seve")

right.release()
left.release()
cv2.destroyAllWindows()
