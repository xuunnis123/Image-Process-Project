# Image-Process-Project
# Stitch, Contours, Dilate, and Erosion applied in Maze
### NCU x 影像處理作業



### Create at  2022.1.15
 
## 介紹
本次Project中，發想透過走迷宮的方式希望能多加應用到影像處理的相關技術，解決平常兩個不同角度但是同場景的圖片接合問題，以及完成使用影像處理的方式完成路徑偵測規劃的目的。
這個Project的處理基本上分成兩個部分，第一部分為影像接合的應用，其中包含SIFT、特徵點找尋、Homography Matrix、以及RANdom SAmple Consensus等處理方式，進行影像接合的處理；結合後的視覺處理，應用了linear blending with constant width進行色差處理。
影像結合完成後，為我們第二部分的應用，針對接合的結果進行應用，包含使用輪廓點找尋、輪廓點繪製、Dilate、Erosion等等相關處理。
透過以上這些技術，最後順利將一開始分成一半的左半部迷宮以及右半部迷宮接合成完整版迷宮，之後再透過找輪廓的方式將路徑透過影像處理的方式完成路徑的繪製完成我們本次Project的目標完成走迷宮的小專案。

## 使用技術

==Stitch實作圖片接合拼接的功能==
1.	特徵點偵測
影像要接合前需要辨別出兩張圖是否有一致性，所以我們要先標出兩張圖的特徵點。
* 使用SIFT進行特徵點偵測。
* SIFT
* 偵測與描述影像中的局部性特徵，它在空間尺度中尋找極值點，並提取出其位置、尺度、旋轉不變數。
* 若找兩影像的關鍵點特徵來比對，可發現越相似的影像，關鍵點特徵符合數越多，藉此來判斷兩影像的相似度。
* SIFT Detector
* 使用課程中DoG (Difference of Gaussian)的觀念處理。

2.	特徵點描述
SIFT Descriptor 進行特徵解釋
* 計算特徵點周圍的梯度方向，用這些梯度來比對不同圖上的特徵點是否相同。
* 將一個特徵點周圍切成4*4的區塊，將每個區塊獨立，對區塊中的每個pixel計算梯度，統計該區塊8個方向的梯度各佔多少的比例，因此我們的每一個特徵點會使用 4*4*8的128維度來描述這個點。
* 程式碼中的keypoints是把所有的keypoints使用CV2的Object存起來。
* Features為每個key points中128維的梯度值，以List的格式儲存。

3.	特徵點對應
* 將前兩個步驟中，左、右兩張圖的每個特徵的特徵梯度一一進行比對，看看兩張圖中的哪些點是一樣的，一對一的方式將他們連線起來。
* 這邊使用的方式K-nearest neighbor algorithm進行對應。
* 特徵差異程度則是利用Euclidean distance進行計算。
* 為了改善SIFT進行一一比對會有小範圍是對的，但是放到大範圍來看明顯是錯誤的情況，這邊使用左邊圖的單一特徵點去找右邊圖最好的點以及第二好的點來比較他們之間的距離比值。
* 如果相除的結果接近1，就直接不使用這個點。
	最後比對如圖所示： 
    ![](https://i.imgur.com/TZANsU5.png)

* 這邊配對出來的點有131的點，但是這些點不能夠全部進行使用，以下會再詳細說明。
4.	畫出特徵點對應的圖
在程式中的drawConnect是進行對應點畫記的功能，這邊將上一步畫出來的特徵點進行對應畫記。
 ![](https://i.imgur.com/3GfNlpK.png)

5.	找出的對應特徵點使用演算法進行計算
* 使用Homography Matrix進行計算，這邊先進行一些名詞解釋
* Homography
* 如果兩台相機拍攝同一場景，兩台相機之間只有旋轉角度的不同，沒有任何位移，則這兩台相機之間的關係稱為 Homography。
* Homography Matrix
* 利用矩陣計算的方式讓左右兩張圖的座標調整成一致的角度。
* 在計算之前會知道剛剛在第三步時是強制讓我們的左邊圖去對應右邊圖的點，但是可能因為拍攝角度的問題，其實左邊圖的點並不存在右邊圖中，造成我們的配對是有問題的。
* 這樣的情況下會造成我們現在要進行轉換時，有了誤差出現，所以這邊要帶入一個隨機抽樣一致演算法( RANdom SAmple Consensus, RANSAC ) 進行轉換前的調整。
* RANSAC
* 利用直線可以共點的方式，可以從一組包含「局外點」的觀測資料集中，透過迭代的方式來估計數學模型的參數。
* 使用一種不確定但是有一定的機率去得出一個合理的結果
* 如果我們要提高機率就可以透過提高迭代次數的方式進行抽樣計算。反覆利用隨機取sample的方式，取出兩個點後當成一條線，並去找出這條線的斜率與截距，這是為了作為等一下代入門檻值時要計算有多少資料點在這個直線的門檻值範圍內。
* 愈多的話最後分數就會愈高，重複執行設定的次數後，取出分數最高的model parameter當成最好的解。
* 透過Threshold（距離）、隨機sample的迭代次數、以及一次隨機sample要多少資料，來調整這個演算法計算的準確度。
* 這邊需要注意的是
* 如果資料量愈大會建議迭代次數要更多。
* Homography在查到的資料中有提到需要使用最少四組pair。
* 執行到最後我們就能夠透過RANSAC的方式來找出在threshold中最多符合的點就是最好的Homography matrix。
* 下圖是我的程式碼調整過程發現準確率滿高的一組參數。 
![](https://i.imgur.com/PLEEvQq.png)

* 最後調整後可以用的點為：26個。 
![](https://i.imgur.com/QOUfd9C.png)


6.	影像拼接
* 延續上一個步驟得到的Homography matrix，最後還需要將右圖的pixel座標轉換到左圖的座標。
* 這邊還有另外一個問題需要解決，因為剛剛的Homography matrix會有浮點數出現，所以Homography matrix會有所誤差造成圖像的缺失。
* 改進方式：將左圖的座標移到右圖找對應點再貼回左圖中來補足缺失的部分，使用inv_H = inv(H), inv_H*p’ = p 來求解，這樣的方式雖然還是有浮點數的值，但是已經能夠使用nearest neighbor 的方法來調整定位。

7.	調整拼接圖色差
* 使用linear blending with constant width，找到兩張圖重疊的區域然後取中間線出來，利用這條中間線左右取固定寬度再做linear blending。




==輪廓點、Dilate、Erosion==
8.	接合完兩張以後，存下接合成的圖後準備走迷宮。
實作完兩張圖分別為左圖跟右圖的接合後，接著呼叫Maze class中的 runMaze()，進行Maze的處理。
9.	將讀進來的地圖改成灰階。
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
因為我們之後只需要使用輪廓分析，所以直接改成灰階比較能確保圖是比較好處理的。
10.	檢測輪廓
* 先將我們的圖二值化。
* 使用cv2中的findContour函式找尋輪廓。 
![](https://i.imgur.com/6EKPrK7.png)

* 這邊只需要使用找外部輪廓就好，所以使用`RETR_EXTERNAL` 。
* 儲存所有輪廓點，所以使用`CHAIN_APPROX_NONE`。
* 這個函式計算完，將會回傳輪廓點與輪廓間的層次，但是我們沒有交疊的情況所以不需要後面這個變數。
* 找到輪廓後，再使用cv2中的drawContour函式進行輪廓繪製。  
![](https://i.imgur.com/HDWg1De.png)
![](https://i.imgur.com/STszbBd.png)

將剛剛的二值圖傳入後，結合剛剛findContour找出來的輪廓點，進行畫圖。最後變成
  (第一個drawContours繪製結果)
  ![](https://i.imgur.com/W5OJTql.png)
(第二個延續上面一張的結果再調整)
	  ![](https://i.imgur.com/yIvg0HM.png)

11.	再做一次二值化，確保是灰階圖，因為待會的Dilate 以及 Erosion是針對白色的部分做處理。
12.	處理卷積
 ![](https://i.imgur.com/hF5gHk0.png)

這邊開始為了等一下的Dilate 以及 Erosion做準備，先準備了一個19x19的卷積kernel。
13.	做Dilate
* 目的：希望藉由Dilate的功能，將我們的白色前景放大，黑色背景縮小。
* 實際作法：如果kernel下的至少一個像素為「1」，則像素元素為「1」，它增加了圖像中的白色區域。
* 程式碼 
![](https://i.imgur.com/7nX3M72.png)

* 結果
 ![](https://i.imgur.com/zCQnbBM.png)

14.	做Erosion
* 目的：依照Kernel的大小，將邊界附近的所有像素都將被丟棄，前景對象的厚度或大小減小，導致圖像中的白色區域減小。
* 實際作法：Kernel在圖像中移動，只有當Kernel下的像素都是1時，原始圖像中的像素（1或0）才會被視為1，否則它將被侵蝕變成0。
* 程式碼 
![](https://i.imgur.com/BQJRvLK.png)

* 結果 
![](https://i.imgur.com/Zz5qB7T.png)

15.	之所以要經過13、14兩個步驟是因為我們想要拿到輪廓邊的變化。
透過第13步先膨脹然後第14步再侵蝕掉的兩張圖，互相比較差異就能夠畫出一張差異圖，而這個差異圖就是我們的路線。
 ![](https://i.imgur.com/SVjjY6r.png)

差異結果 
![](https://i.imgur.com/PFki9Oj.png)

16.	將得到的結果應用回去原圖並輸出。
* 當畫出第15步驟的路徑圖以後代表我們成功找到一條讓這個迷宮分成兩半的圖，而這個就是我們的迷宮正確走法。
*  這時需要將這個路徑的圖，與我們原圖進行疊層輸出成結果。
甲、	將原圖拆成r, g, b三個項目。
乙、	將第15步驟的圖反轉1->0、0->1，做成遮罩。
    ![](https://i.imgur.com/ndUL2jJ.png)

丙、	透過以下程式碼，在r, b兩個channels中套上遮罩
     ![](https://i.imgur.com/7BzZBBA.png)
    ![](https://i.imgur.com/SrjrVpb.png)
    ![](https://i.imgur.com/EwHhMp1.png)


  
Channel g的部分不需套上這個遮罩，因為我希望最後使用綠色來畫出正確路徑。
 ![](https://i.imgur.com/47DNXKZ.png)

17.	合併完成迷宮，結果輸出見下個段落。

## 結果
本次作業希望能透過兩張不同邊的地圖進行接合，然後透過影像處理的相關技巧，進行迷宮路徑規劃。結果也如預期最終有順利將左邊圖跟右邊圖進行接合，並且最終透過輪廓分析等等方式，畫出迷宮的路徑來，透過以下幾張代表性圖片完整呈現整的Project想要達成的目的。
1.	一開始的輸入為相同大小的兩張部分迷宮圖
左半邊迷宮圖
 ![](https://i.imgur.com/wUfB0pY.png)

右半邊迷宮圖
 ![](https://i.imgur.com/xn1Wbfo.png)

2.	影像接合完成
 ![](https://i.imgur.com/MRnpbsA.png)


3.	將接合過的影像，經過Dilate與Erosion畫出差異路徑   
![](https://i.imgur.com/DrHD7z4.png)
![](https://i.imgur.com/gQN2nJ9.png)
![](https://i.imgur.com/A0BvFRT.png)


4.	最終經過處理後得到標上顏色路徑的結果圖
 ![](https://i.imgur.com/6brJhHx.png)







## 討論/結論

經過這次的影像處理作業，對於影像接合以及其中上課提到的演算法更加了解，雖然有些數學理論不太確定怎麼推導，但是再盡力了解與參考網路上一些教學後，看到能成功做出接合的動作，還滿有收穫的。
不過有出現偶爾還是會不準的情況，如下圖，圖片接合時出現歪掉的情況，自己設想應該是RANSAC那一塊出問題，可能在隨機抓取的時候出現誤差。目前遇到這個問題的時候，是直接重新執行第二次就能成功得到所要的結果了。
 ![](https://i.imgur.com/y4fd0Nf.png)
![](https://i.imgur.com/4KW02W6.png)

 

而後續將左圖以及右圖接合後產出的圖，再持續使用影像處理中輪廓尋找、繪製以及Dilate與Erosion等功能，透過比較差異的方式，找出迷宮中能夠切成兩塊輪廓而這個輪廓，正是我們正確的迷宮道路，這個觀念過去沒有想到過可以應用在這裡，往往想到走迷宮都是想說可能要使用DFS或BFS的方式進行。在這個Project進行的過程中，也再次複習到很多影像處理上課的理論，在應用的過程中，雖然沒辦法做到完全手刻，但是對於理論與應用也更加了解。後續希望能夠找尋出有時候接合會失準的原因，再持續調整。



## Reference
[Camera Calibration相機校正. 攝像機標定(Camera… | by Leyan Bin Veon | Digital Image Recognition & ML Note | Medium](https://medium.com/image-processing-and-ml-note/camera-calibration%E7%9B%B8%E6%A9%9F%E6%A0%A1%E6%AD%A3-1d94ffa7cbb4)
	[[轉] 隨機抽樣一致性算法（RANSAC） @ Rocky的部落格 :: 痞客邦 :: (pixnet.net)](https://rocky69.pixnet.net/blog/post/218271061)
	[Python-OpenCV 19. 圖像處理形態學變換](https://kknews.cc/code/63qzkxl.html)

