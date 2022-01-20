

import numpy as np
import matplotlib.pyplot as plt
import cv2
import random

class Stitcher:
    def __init__(self):
        pass
    
    def stitch(self, imgs, blending_mode = "linearBlending", ratio = 0.75):
        
        
        img_left, img_right = imgs
        (left_h, left_width) = img_left.shape[:2]
        (right_h, right_width) = img_right.shape[:2]
        print("Left img size (", left_h, "*", left_width, ")")
        print("Right img size (", right_h, "*", right_width, ")")
        
        # Step1 - extract the keypoints and features by SIFT detector and descriptor
        print("Step1 - Extract the keypoints and features by SIFT detector and descriptor...")
        left_keypoints, left_features = self.detectEdges(img_left)
        right_keypoints, right_features = self.detectEdges(img_right)
        

        # Step2 - extract the match point with threshold
        print("Step2 - Extract the match point with threshold (David Lowe’s ratio test)...")
        position_match = self.matchKeypoint(left_keypoints, right_keypoints, left_features, right_features, ratio)
        print("The number of matching points:", len(position_match))
        
        # Step2 - draw the img with matching point and their connection line
        vis = self.matchConnect([img_left, img_right])
        self.drawConnect(position_match, left_width, vis)
        
        # Step3 - fit the homography model with RANSAC algorithm
        print("Step3 - Fit the best homography model with RANSAC algorithm...")
        model_match = self.fitModel(position_match)
        
    
        # Step4 - Warp image to create panoramic image
        print("Step4 - Warp image to create panoramic image...")
        warp_img = self.warp([img_left, img_right], model_match, blending_mode) 
        
        return warp_img
    
    def detectEdges(self, img):
        '''
        The Detector and Descriptor
        '''
        # SIFT detector and descriptor
        sift = cv2.xfeatures2d.SIFT_create()
        keypoints, features = sift.detectAndCompute(img,None)
        print("keypoints=",keypoints)
        print("features=",features)
        return keypoints, features
    
    def matchKeypoint(self, left_keypoints, right_keypoints, left_features, right_features, ratio):
        
        match_index_and_distance = [] # min corresponding index, min distance, seccond min corresponding index, second min distance
        for i in range(len(left_features)):
            min_distance = [-1, np.inf]  # record the min corresponding index, min distance
            
            sec_min_distance = [-1 ,np.inf]  # record the second corresponding min index, min distance
            
            for j in range(len(right_features)):
                distance = np.linalg.norm(left_features[i] - right_features[j])
                if (min_distance[1] > distance):
                    sec_min_distance = np.copy(min_distance)
                    min_distance = [j , distance]
                    
                elif (sec_min_distance[1] > distance and sec_min_distance[1] != min_distance[1]):
                    sec_min_distance = [j, distance]
                    
            match_index_and_distance.append([min_distance[0], min_distance[1], sec_min_distance[0], sec_min_distance[1]])
            
     
        goodMatches = []
        for i in range(len(match_index_and_distance)):
            if (match_index_and_distance[i][1] <= match_index_and_distance[i][3] * ratio):
                goodMatches.append((i, match_index_and_distance[i][0]))
            
        good_position_match = []
        for (idx, correspondingIdx) in goodMatches:
            psA = (int(left_keypoints[idx].pt[0]), int(left_keypoints[idx].pt[1]))
            psB = (int(right_keypoints[correspondingIdx].pt[0]), int(right_keypoints[correspondingIdx].pt[1]))
            good_position_match.append([psA, psB])
            
        return good_position_match
    
    def matchConnect(self, imgs):
       
        
        # initialize the output visualization image
        img_left, img_right = imgs
        (left_h, left_width) = img_left.shape[:2]
        (right_h, right_width) = img_right.shape[:2]
        vis = np.zeros((max(left_h, right_h), left_width + right_width, 3), dtype="uint8")
        vis[0:left_h, 0:left_width] = img_left
        vis[0:right_h, left_width:] = img_right
        
       
        return vis
    
    def drawConnect(self,position_match,left_width,vis):
        for (img_left_pos, img_right_pos) in position_match:

                pos_l = img_left_pos
                pos_r = img_right_pos[0] + left_width, img_right_pos[1]
                cv2.circle(vis, pos_l, 3, (0, 0, 255), 1)
                cv2.circle(vis, pos_r, 3, (0, 255, 0), 1)
                cv2.line(vis, pos_l, pos_r, (255, 0, 0), 1)
                
        # return the visualization
        
        cv2.imwrite("img/matching.jpg", vis)
        
        return vis
        
    def fitModel(self, position_match):
        
        dstPoints = [] # i.e. left image(destination image)
        srcPoints = [] # i.e. right image(source image) 
        for dstPoint, srcPoint in position_match:
            print("fitmodel=",dstPoint,srcPoint)
            dstPoints.append(list(dstPoint)) 
            srcPoints.append(list(srcPoint))
        dstPoints = np.array(dstPoints)
        srcPoints = np.array(srcPoints)
        
        homography = Homography()
        
        # RANSAC algorithm (RANdom SAmple Consensus，RANSAC), selecting the best fit homography
        NumSample = len(position_match)
        threshold = 5.0  
        NumIter = 7000  
        NumRamdomSubSample = 4 
        MaxInlier = 0
        best = None
        
        for run in range(NumIter):
            SubSampleIdx = random.sample(range(NumSample), NumRamdomSubSample) # get the Index of ramdom sampling
            H = homography.solve_homography(srcPoints[SubSampleIdx], dstPoints[SubSampleIdx])
            
            # find the best Homography have the the maximum number of inlier
            NumInlier = 0 
            for i in range(NumSample):
                if i not in SubSampleIdx:
                    concateCoor = np.hstack((srcPoints[i], [1])) # add z-axis as 1
                    dstCoor = H @ concateCoor.T # calculate the coordination after transform to destination img 
                    if dstCoor[2] <= 1e-8: # avoid divide zero number, or too small number cause overflow
                        continue
                    dstCoor = dstCoor / dstCoor[2]
                    if (np.linalg.norm(dstCoor[:2] - dstPoints[i]) < threshold):
                        NumInlier = NumInlier + 1
            if (MaxInlier < NumInlier):
                MaxInlier = NumInlier
                best = H
                
        print("The Number of Maximum Inlier:", MaxInlier)
        
        return best
    
    def warp(self, imgs, model_match, blending_mode):
        
        img_left, img_right = imgs
        (left_h, left_width) = img_left.shape[:2]
        (right_h, right_width) = img_right.shape[:2]
        stitch_img = np.zeros( (max(left_h, right_h), left_width + right_width, 3), dtype="int") # create the (stitch)big image accroding the imgs height and width 
        
        if (blending_mode == "noBlending"):
            stitch_img[:left_h, :left_width] = img_left
            
        # Transform Right image(the coordination of right image) to destination iamge(the coordination of left image) with model_match
        inv_H = np.linalg.inv(model_match)
        for i in range(stitch_img.shape[0]):
            for j in range(stitch_img.shape[1]):
                coor = np.array([j, i, 1])
                img_right_coor = inv_H @ coor # the coordination of right image
                img_right_coor /= img_right_coor[2]
                
                # you can try like nearest neighbors or interpolation  
                y, x = int(round(img_right_coor[0])), int(round(img_right_coor[1])) # y for width, x for height
                
                
                # if the computed coordination not in the (hegiht, width) of right image, it's not need to be process 
                if (x < 0 or x >= right_h or y < 0 or y >= right_width):
                    continue
                # else we need the tranform for this pixel
                stitch_img[i, j] = img_right[x, y]
            
        
        # create the Blender object to blending the image
        
       
        if (blending_mode == "linearBlendingWithConstant"):
             
            
            img_left, img_right = [img_left, stitch_img]
            (left_h, left_width) = img_left.shape[:2]
            (right_h, right_width) = img_right.shape[:2]
            img_left_mask = np.zeros((right_h, right_width), dtype="int")
            img_right_mask = np.zeros((right_h, right_width), dtype="int")
            constant_width = 3 # constant width
            
            # find the left image and right image mask region(Those not zero pixels)
            for i in range(left_h):
                for j in range(left_width):
                    if np.count_nonzero(img_left[i, j]) > 0:
                        img_left_mask[i, j] = 1
            for i in range(right_h):
                for j in range(right_width):
                    if np.count_nonzero(img_right[i, j]) > 0:
                        img_right_mask[i, j] = 1
                        
            # find the overlap mask(overlap region of two image)
            overlap_mask = np.zeros((right_h, right_width), dtype="int")
            for i in range(right_h):
                for j in range(right_width):
                    if (np.count_nonzero(img_left_mask[i, j]) > 0 and np.count_nonzero(img_right_mask[i, j]) > 0):
                        overlap_mask[i, j] = 1
            
            # compute the alpha mask to linear blending the overlap region
            alpha_mask = np.zeros((right_h, right_width)) # alpha value depend on left image
            for i in range(right_h):
                minIdx = maxIdx = -1
                for j in range(right_width):
                    if (overlap_mask[i, j] == 1 and minIdx == -1):
                        minIdx = j
                    if (overlap_mask[i, j] == 1):
                        maxIdx = j
                
                if (minIdx == maxIdx): # represent this row's pixels are all zero, or only one pixel not zero
                    continue
                    
                decrease_step = 1 / (maxIdx - minIdx)
                
                # Find the middle line of overlapping regions, and only do linear blending to those regions very close to the middle line.
                middleIdx = int((maxIdx + minIdx) / 2)
                
                # left 
                for j in range(minIdx, middleIdx + 1):
                    if (j >= middleIdx - constant_width):
                        alpha_mask[i, j] = 1 - (decrease_step * (j - minIdx))
                    else:
                        alpha_mask[i, j] = 1
                # right
                for j in range(middleIdx + 1, maxIdx + 1):
                    if (j <= middleIdx + constant_width):
                        alpha_mask[i, j] = 1 - (decrease_step * (j - minIdx))
                    else:
                        alpha_mask[i, j] = 0

            
            linearBlendingWithConstantWidth_img = np.copy(img_right)
            linearBlendingWithConstantWidth_img[:left_h, :left_width] = np.copy(img_left)
            # linear blending with constant width
            for i in range(right_h):
                for j in range(right_width):
                    if (np.count_nonzero(overlap_mask[i, j]) > 0):
                        linearBlendingWithConstantWidth_img[i, j] = alpha_mask[i, j] * img_left[i, j] + (1 - alpha_mask[i, j]) * img_right[i, j]
            
            stitch_img = linearBlendingWithConstantWidth_img

        
        # remove the black border
        
        h, w = stitch_img.shape[:2]
        reduced_h, reduced_w = h, w
        # right to left
        for col in range(w - 1, -1, -1):
            all_black = True
            for i in range(h):
                if (np.count_nonzero(stitch_img[i, col]) > 0):
                    all_black = False
                    break
            if (all_black == True):
                reduced_w = reduced_w - 1
                
        # bottom to top 
        for row in range(h - 1, -1, -1):
            all_black = True
            for i in range(reduced_w):
                if (np.count_nonzero(stitch_img[row, i]) > 0):
                    all_black = False
                    break
            if (all_black == True):
                reduced_h = reduced_h - 1
        
        stitch_img = stitch_img[:reduced_h, :reduced_w]
        
        return stitch_img
   


class Homography:
    def solve_homography(self, source_plane, target_plane):
        
        try:
            A = []  
            for r in range(len(source_plane)): 
                #print(target_plane[r, 0])
                A.append([-source_plane[r,0], -source_plane[r,1], -1, 0, 0, 0, source_plane[r,0]*target_plane[r,0], source_plane[r,1]*target_plane[r,0], target_plane[r,0]])
                A.append([0, 0, 0, -source_plane[r,0], -source_plane[r,1], -1, source_plane[r,0]*target_plane[r,1], source_plane[r,1]*target_plane[r,1], target_plane[r,1]])

            u, s, vt = np.linalg.svd(A) # Solve s ystem of linear equations Ah = 0 using SVD
            # pick H from last line of vt  
            H = np.reshape(vt[8], (3,3))
            # normalization, let H[2,2] equals to 1
            H = (1/H.item(8)) * H
        except:
            print("Error occur!")

        return H


class Maze:
    def runMaze(self):

        img = cv2.imread('img/mix_to_maze.jpg')


        print("Step5 - Run the Maze...")
        # Binary conversion
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # Inverting tholdolding will give us a binary image with a white wall and a black background.
        ret, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV) 

        # Contours 檢測輪廓
        contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        # 畫出黑白的輪廓
        dc = cv2.drawContours(thresh, contours, 0, (255, 255, 255), 5) #0
        cv2.imshow('Contours 1', dc)
        save=dc
        dc = cv2.drawContours(dc, contours, 1, (0,0,0) , 5)

        cv2.imshow('Contours 2', dc)
    
        # 二值化處理
        ret, thresh = cv2.threshold(dc, 240, 255, cv2.THRESH_BINARY)

        # 卷積大小 19x19
        ke = 19
        kernel = np.ones((ke, ke), np.uint8) 

        # Dilate 影像膨脹
       

        dilation = cv2.dilate(thresh, kernel, iterations=1)
        cv2.imshow('dilation', dilation)

        # Erosion 影像侵蝕
        
       
        erosion = cv2.erode(dilation, kernel, iterations=1)
        cv2.imshow('erosion', erosion)
        # Find differences between two
        # 前景\背景分離
        diff = cv2.absdiff(dilation, erosion)
        cv2.imshow('diff', diff)

        # splitting the channels of maze
        b, g, r = cv2.split(img)


        mask_inv = cv2.bitwise_not(diff)

        cv2.imshow('mask_inv', mask_inv)
        r = cv2.bitwise_and(r, r, mask=mask_inv)
        b = cv2.bitwise_and(b, b, mask=mask_inv)
        
        cv2.imshow('r ', r)
        cv2.imshow('b ', b)
        cv2.imshow('g', g)
        res = cv2.merge((b, g, r)) 
       
        cv2.imshow('Solved Maze', res)
        
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
if __name__ == "__main__":
    
    fileNameList = [('left', 'right')]
    for fname1, fname2 in fileNameList:
        # Read the img file
        src_path = "img/"
        fileName1 = fname1
        fileName2 = fname2
        img_left = cv2.imread(src_path + fileName1 + ".jpg")
        img_right = cv2.imread(src_path + fileName2 + ".jpg")
        
        # The stitch object to stitch the image
        blending_mode = "linearBlendingWithConstant" # three mode - noBlending、linearBlending、linearBlendingWithConstant
        stitcher = Stitcher()
        warp_img = stitcher.stitch([img_left, img_right], blending_mode)

        # plot the stitched image
        plt.figure(13)
        plt.title("warp_img")
        plt.imshow(warp_img[:,:,::-1].astype(int))

        # save the stitched iamge
        saveFilePath = "img/mix_to_maze.jpg".format(fileName1, fileName2, blending_mode)
        cv2.imwrite(saveFilePath, warp_img)
    
    maze = Maze()
    maze.runMaze()
