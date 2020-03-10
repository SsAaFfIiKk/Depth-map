import cv2
import numpy as np


imgL = cv2.imread("1_l.png")
imgR = cv2.imread("1_r.png")
# imgL = cv2.imread("2_l.png")
# imgR = cv2.imread("2_r.png")


sift = cv2.xfeatures2d.SIFT_create()

kpL, desL = sift.detectAndCompute(imgL, None)
kpR, desR = sift.detectAndCompute(imgR, None)

FLANN_INDEX_KDTREE = 1
index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
search_params = dict(checks=50)
flann = cv2.FlannBasedMatcher(index_params,search_params)
matches = flann.knnMatch(desL, desR, k=2)
good = []
ptsL = []
ptsR = []

for i,(m,n) in enumerate(matches):
    if m.distance < 0.8*n.distance:
        good.append(m)
        ptsR.append(kpR[m.trainIdx].pt)
        ptsL.append(kpL[m.queryIdx].pt)

ptsL = np.int32(ptsL)
ptsR = np.int32(ptsR)
F, mask = cv2.findFundamentalMat(ptsL, ptsR, cv2.FM_LMEDS)

ptsL = ptsL[mask.ravel() == 1]
ptsR = ptsR[mask.ravel() == 1]

def drawlines(imgL, imgR, lines, ptsL, ptsR):
    (r, c) = imgL.shape[:2]
    imgL = cv2.cvtColor(imgL, cv2.COLOR_BGR2GRAY)
    imgR = cv2.cvtColor(imgR, cv2.COLOR_BGR2GRAY)
    for r,pt1,pt2 in zip(lines, ptsL, ptsR):
        color = tuple(np.random.randint(0,255,3).tolist())
        x0,y0 = map(int, [0, -r[2]/r[1] ])
        x1,y1 = map(int, [c, -(r[2]+r[0]*c)/r[1] ])
        imgL = cv2.line(imgL, (x0, y0), (x1, y1), color, 1)
        imgL = cv2.circle(imgL, tuple(pt1), 5, color, -1)
        imgR = cv2.circle(imgR, tuple(pt2), 5, color, -1)
    return imgL, imgR


lines1 = cv2.computeCorrespondEpilines(ptsR.reshape(-1, 1, 2), 2, F)
lines1 = lines1.reshape(-1, 3)
img5, img6 = drawlines(imgL, imgR, lines1, ptsL, ptsR)

lines2 = cv2.computeCorrespondEpilines(ptsL.reshape(-1, 1, 2), 1, F)
lines2 = lines2.reshape(-1, 3)
img3, img4 = drawlines(imgL, imgR, lines2, ptsR, ptsL)

cv2.imshow("", np.hstack([img5, img3]))
cv2.waitKey(0)
