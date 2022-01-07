import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras
from math import floor

def four_corners_sort(pts):
    diff = np.diff(pts, axis=1)
    summ = pts.sum(axis=1)
    return np.float32([pts[np.argmin(summ)], pts[np.argmax(diff)],
                    pts[np.argmax(summ)], pts[np.argmin(diff)]])

def prediction(num_img):
    num_img= cv2.resize(num_img,(28,28))
    num_img= num_img.reshape(1,28,28,1)
    num_img= num_img/255.
    pred= model.predict(num_img)
    confidence= np.amax(pred)/np.sum(pred,1)*100
    return np.argmax(pred), confidence

def heuristic(queue,ex,ey):
    queue= np.array(queue)
    queue= queue-[ex,ey]
    queue= queue**2
    dist= np.sum(queue,axis=1)
    ind= np.argmin(dist)
    return ind

def correction(i,j,l,m):
    i= round((i-15)/30)*30 +15
    j= round((j-15)/30)*30 +15
    l= round((l-15)/30)*30 +15
    m= round((m-15)/30)*30 +15
    return i,j,l,m

def backtrack(dirxn,img, img1,ex,ey,incr,incr1,ix,iy):
    color= (np.random.randint(20,200),np.random.randint(20,200),np.random.randint(20,200))
    i,j=ex,ey
    while True:
        if dirxn[int((j-incr)/incr1)][int((i-incr)/incr1)]==1:
            img1= cv2.line(img1,(i,j),(i,j+incr1),color,2)
            img= cv2.line(img,(i,j),(i,j+incr1),color,2)
            j+=incr1
        elif dirxn[int((j-incr)/incr1)][int((i-incr)/incr1)]==2:
            img1= cv2.line(img1,(i,j),(i-incr1,j),color,2)
            img= cv2.line(img,(i,j),(i-incr1,j),color,2)
            i-=incr1
        elif dirxn[int((j-incr)/incr1)][int((i-incr)/incr1)]==3:
            img1= cv2.line(img1,(i,j),(i,j-incr1),color,2)
            img= cv2.line(img,(i,j),(i,j-incr1),color,2)
            j-=incr1
        elif dirxn[int((j-incr)/incr1)][int((i-incr)/incr1)]==4:
            img1= cv2.line(img1,(i,j),(i+incr1,j),color,2)
            img= cv2.line(img,(i,j),(i+incr1,j),color,2)
            i+=incr1
        if ix==i and iy==j:
            break
    return img,img1

def a_star(start,end,temp,thresh,img,iteration=0):
    incr=15
    incr1=2*incr
    c=0
    img1=temp.copy()
    ix,iy= start
    ex,ey= end
    ix,iy,ex,ey= correction(ix,iy,ex,ey) 
    m= thresh.shape[0]
    n= thresh.shape[1]
    if not thresh[ey][ex]==255:
        thresh[ey][ex]=255
    queue=[[ix,iy]]
    i,j=ix,iy
    dirxn=np.zeros((int(m/incr1),int(n/incr1)),np.uint8)
    print(ix,iy,ex,ey)
    while True:
        index= heuristic(queue,ex,ey)
        i,j= queue[index]
        queue.remove([i,j])
        thresh[j][i]=0
    
        if 0<=(j-incr1)<m and thresh[j-incr][i]>=127 and thresh[j-incr1][i]>=127:
            thresh= cv2.circle(thresh,(i,j-incr1),2,120,2)
            queue.append([i,j-incr1]) 
            dirxn[floor(j/incr1)-1][floor(i/incr1)]= 1
            if (ey-c)<=(j-incr1)<=(ey+c) and (ex-c)<= i<=(ex+c):
                break
        if 0<=(j+incr1)<m and thresh[j+incr][i]>=127 and thresh[j+incr1][i]>=127 :
            thresh= cv2.circle(thresh,(i,j+incr1),2,120,2)
            queue.append([i,j+incr1]) 
            dirxn[floor(j/incr1)+1][floor(i/incr1)]= 3
            if (ey-c)<=(j+incr1)<=(ey+c) and (ex-c)<=i<=(ex+c):
                break
        if 0<=(i-incr1)<n and thresh[j][i-incr]>=127 and thresh[j][i-incr1]>=127:
            thresh= cv2.circle(thresh,(i-incr1,j),2,120,2)
            queue.append([i-incr1,j]) 
            dirxn[floor(j/incr1)][floor(i/incr1)-1]=4
            if (ey-c)<=j<=(ey+c) and (ex-c)<=(i-incr1)<=(ex+c):
                break
        if 0<=(i+incr1)<n and thresh[j][i+incr]>=127 and thresh[j][i+incr1]>=127:
            thresh= cv2.circle(thresh,(i+incr1,j),2,120,2)
            queue.append([i+incr1,j]) 
            dirxn[floor(j/incr1)][floor(i/incr1)+1]=2
            if (ey-c)<=j<=(ey+c) and (ex-c)<= (i+incr1)<=(ex+c):
                break
            
        cv2.imshow('intermediate',thresh)
        if c==0:
            cv2.waitKey(3000)
            c=1
        cv2.waitKey(50)
        
    print('Found Path')
    img,img1= backtrack(dirxn,img,img1,ex,ey,incr,incr1,ix,iy)
    cv2.imshow('path'+str(iteration),img1)
    return img

model= keras.models.load_model('model7.h5')

img1= cv2.imread('maze.jpg')
img1= cv2.resize(img1,(500,500))

gray= cv2.cvtColor(img1,cv2.COLOR_BGR2GRAY)
_,thresh= cv2.threshold(gray,200,255,cv2.THRESH_BINARY_INV)
kernel= np.ones((5,5),np.uint8)
thresh= cv2.dilate(thresh,kernel,iterations=3)
thresh= cv2.erode(thresh,kernel,iterations=2)
contour,_= cv2.findContours(thresh,cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
for cnt in contour:
    approx= cv2.approxPolyDP(cnt,0.01*cv2.arcLength(cnt,True),True)
    break
cv2.imshow('Original',img1)
approx = np.reshape(approx, (approx.shape[0] * approx.shape[1], approx.shape[2]))
c= four_corners_sort(approx)      #np.float32(approx)
crop= np.float32([[0,0],[0,600],[600,600],[600,0]])
M= cv2.getPerspectiveTransform(c,crop)
img1= cv2.warpPerspective(img1,M,(600,600))
#### code for finding the digits starts here....
visits=[]
hsv= cv2.cvtColor(img1,cv2.COLOR_BGR2HSV)
lr=np.array([0,24,131])
ur=np.array([255,255,255])
mask= cv2.inRange(hsv,lr,ur)
contours,_= cv2.findContours(mask,cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
img= img1.copy()
digit=np.zeros((600,600,3),np.uint8)
for cnt in contours:
    if cv2.contourArea(cnt)>40:
        x,y,w,h = cv2.boundingRect(cnt)
        visits.append([int(x+w/2),int(y+h/2)])
        num_img= mask[y-5:y+h+5,x-5:x+w+5]
        digit[y-5:y+h+5,x-5:x+w+5]= img[y-5:y+h+5,x-5:x+w+5]
        img[y:y+h,x:x+w]= [255,255,255]
        num, conf= prediction(num_img)
        text= 'Prediction-'+str(num)
        text2='Confidence%-'+str(conf)
        print(text+text2)
        cv2.putText(digit,text,(x-40,y-20),cv2.FONT_HERSHEY_SIMPLEX,0.4,(200,255,255))
        cv2.putText(digit,text2,(x-40,y+h+20),cv2.FONT_HERSHEY_SIMPLEX,0.4,(200,255,255))

cv2.imshow('digits',digit)
#### code ends here.....

gray= cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
exits=[]
pt=0
_,thresh= cv2.threshold(gray,127,255,cv2.THRESH_BINARY)
thresh= cv2.erode(thresh,np.ones((7,7),np.uint8),iterations=2)
sampling= int(600/40)

#### code for finding exits starts here.....
for j in range(0,4,1):
    for i in range(15,600,sampling):
        if pt==2:
            break
        if j==0 and thresh[0][i]==255:
           exits.append([i,15])
           pt+=1
           img1= cv2.circle(img1,(i,5),2,(0,250,0),5)
        elif j==1 and thresh[599][i]==255:
            exits.append([i,585])
            pt+=1
            img1= cv2.circle(img1,(i,594),2,(0,250,0),5)
        elif j==2 and thresh[i][0]==255:
            exits.append([15,i])
            pt+=1
            img1= cv2.circle(img1,(5,i),2,(0,250,0),5)
        elif j==3 and thresh[i][599]==255:
            exits.append([585,i])
            pt+=1
            img1= cv2.circle(img1,(594,i),2,(0,250,0),5)

print('Exits are found succesfully and the points are>>>'+str(exits[0])+'and'+str(exits[1]))
#### code ends here......

#### code for arranging order of visits starts here.........
i,j=exits[0]
points=[[i,j]]
while len(visits):
    ind= heuristic(visits,i,j)
    points.append(visits[ind])
    i,j= visits[ind]
    visits.remove(visits[ind])
points.append(exits[1])
#### code ends here........

temp= img1.copy()
for i in range(0,len(points)-1):
    thresh_copy= thresh.copy()
    img1=a_star(points[i],points[i+1],temp,thresh_copy,img1,i)    
###cv2.imshow('ALL Path',img1)

cv2.waitKey(0)
cv2.destroyAllWindows()