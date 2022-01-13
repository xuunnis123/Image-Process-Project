import cv2
import numpy as np













img = cv2.imread('img/u4.jpg')



# Binary conversion
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# Inverting tholdolding will give us a binary image with a white wall and a black background.
ret, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV) 

# Contours 檢測輪廓
contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
# 畫出黑白的輪廓
dc = cv2.drawContours(thresh, contours, 0, (255, 255, 255), 5)
cv2.imshow('Contours 1', dc)

dc = cv2.drawContours(dc, contours, 1, (0,0,0) , 5)

cv2.imshow('Contours 2', dc)
# 二值化處理
ret, thresh = cv2.threshold(dc, 240, 255, cv2.THRESH_BINARY)

# 卷積大小 19x19
ke = 19
kernel = np.ones((ke, ke), np.uint8) 

# Dilate 影像膨脹
'''
Dilation is one of the two basic operators in the field of mathematical morphology, and the other is erosion.
It is usually applied to binary images, but there are some versions available for grayscale images.
The basic effect of the operator on binary images is to gradually enlarge the boundaries of foreground pixel regions (usually white pixels).
Therefore, the size of the foreground pixel increases, and the holes in these areas become smaller.
'''

dilation = cv2.dilate(thresh, kernel, iterations=1)
cv2.imshow('dilation', dilation)

# Erosion 影像侵蝕
'''
Erosion is a form of the second operator.
It also applies to binary images.
The basic effect of the operator on binary images is to eliminate the boundaries of foreground pixel areas (usually white pixels).
Therefore, the area of ​​foreground pixels is reduced, and the holes in these areas become large.
'''
# 第一個參數為二值化的影像 /第二個參數為使用的捲積 kernel / 第三個參數為迭代次數(預設為1)
erosion = cv2.erode(dilation, kernel, iterations=1)
cv2.imshow('erosion', erosion)
# Find differences between two
# 前景\背景分離
diff = cv2.absdiff(dilation, erosion)
cv2.imshow('diff', diff)

# splitting the channels of maze
b, g, r = cv2.split(img)


mask_inv = cv2.bitwise_not(diff)

# In order to display the solution on the original maze image, first divide the original maze into r, g, b components.
# Now create a mask by inverting the diff image.
# The bitwise and r and g components of the original maze using the mask created in the last step.
# This step will remove the red and green components from the image portion of the maze solution.
# The last one is to merge all the components and we will use the blue marked solution.

# masking out the green and red colour from the solved path
r = cv2.bitwise_and(r, r, mask=mask_inv)
b = cv2.bitwise_and(b, b, mask=mask_inv)

res = cv2.merge((b, g, r))

cv2.imshow('Solved Maze', res)
cv2.waitKey(0)
cv2.destroyAllWindows()