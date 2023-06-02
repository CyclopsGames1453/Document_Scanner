import cv2
import numpy as np

#read image
#the higher resolution slower the processes
image = cv2.imread("image/image.jpg")

#Get width and height values
heightImg = image.shape[0]
withImg = image.shape[1]

#makes the image ready for scanning
def preProcessing(img):
    imgGray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    imgBlur = cv2.GaussianBlur(imgGray,(5,5),1)
    imgCanny = cv2.Canny(imgBlur,200,200)
    kernel = np.ones((5,5))
    imgDial = cv2.dilate(imgCanny,kernel,iterations=2)
    imgThers = cv2.erode(imgDial,kernel,iterations=1)

    return imgThers

#detects the corners of the picture
def getContours(img):
    biggest = np.array([])
    maxArea = 0
    countours,hierarchy = cv2.findContours(img,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
    x,y,w,h = 0,0,0,0
    for cnt in countours:
        area = cv2.contourArea(cnt)
        if area>2000:
            #cv2.drawContours(imgContour,cnt,-1,(255,0,0),3)
            peri=cv2.arcLength(cnt,True)
            approx = cv2.approxPolyDP(cnt,0.02*peri,True)
            if area > maxArea and len(approx) == 4:
                biggest = approx
                maxArea = area
    cv2.drawContours(imgContour,biggest,-1,(255,0,0),10)
    return biggest

#Reorder the corners of the document
def reorder(myPoints):
    myPoints = myPoints.reshape((4,2))
    NewMyPoints = np.zeros((4,1,2),np.int32)
    add=myPoints.sum(1)
    NewMyPoints[0] = myPoints[np.argmin(add)]
    NewMyPoints[3] = myPoints[np.argmax(add)]
    diff=np.diff(myPoints,axis=1)
    NewMyPoints[1] = myPoints[np.argmin(diff)]
    NewMyPoints[2] = myPoints[np.argmax(diff)]

    return NewMyPoints

# crop image
def getWarp(img,biggest):
    biggest = reorder(biggest)
    pts1 = np.float32(biggest)
    pts2 = np.float32([[0,0],[withImg,0],[0,heightImg],[withImg,heightImg]])
    matrix=cv2.getPerspectiveTransform(pts1,pts2)
    imgOutput=cv2.warpPerspective(img,matrix,(withImg,heightImg))

    imgCroppes = imgOutput[10:imgOutput.shape[0]-10,10:imgOutput.shape[1]-10]

    return imgCroppes

# create window
cv2.namedWindow("frame")

# identify corners
imgContour = image.copy()
imgThres = preProcessing(imgContour)
biggest = getContours(imgThres)

# crop image
imageWarped = getWarp(image,biggest)

# show result
cv2.imshow("image",image)
cv2.imshow("frame",imageWarped)

k=cv2.waitKey(0)

# show result
if k == ord("q"):
    print("q key pressed , picture saved.")
    cv2.imwrite("image/result.jpg",imageWarped)

cv2.destroyAllWindows()
