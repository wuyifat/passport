import numpy as np
import cv2
import matplotlib.pyplot as plt

def patchDist(p1,p2):
	# p1, p2 are patches, dim = n*m*3
	return pixelDist(aveColor(p1), aveColor(p2))

def pixelDist(p1,p2):
	# p1, p2 are pixels, dim = 3*1
	return np.sum(np.absolute(p1-p2))

def aveColor(p):
	# p is a patch, dim = n*m*3
	return np.mean(np.mean(p,0),0)

def isBG(x,y,img):
	return pixelDist(img[y,x], bg) < threshold

def isInBG(x,y,img):
	# return true is pixel[y,x] is in background. Useful when looking at random hair
	x1 = max(0, x-5)
	x2 = min(img.shape[1], x+5)
	y1 = max(0, y-5)
	y2 = min(img.shape[0], y+5)
	patch = img[y1:y2, x1:x2]
	return pixelDist(aveColor(patch), bg) < threshold

img = cv2.imread('../img/vc.jpg')
threshold = 50
bg = [255,255,255]
#img = cv2.imread('../img/vc.jpg')
#print img.shape

#img = cv2.resize(img,(540,960))
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

xmlPath = '/home/wuyi/Apps/opencv/data/haarcascades/'
face_cascade = cv2.CascadeClassifier(xmlPath + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(xmlPath + 'haarcascade_eye.xml')
faces = face_cascade.detectMultiScale(gray, 1.3, 3)

print len(faces)
print "********************"

for (x,y,w,h) in faces:
    #cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
    #roi_gray = gray[y:y+h, x:x+w]
    #roi_color = img[y:y+h, x:x+w]
    #print x,y
    #print x+w,y+h

#    inlay = img.crop((x,y,x+w,y+h)).rotate(90)
#    img.paste(inlay,(x,y,x+w,y+h))
    
    #eyes = eye_cascade.detectMultiScale(roi_gray)
    #for (ex,ey,ew,eh) in eyes:
    #	cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)

    top_left = img[0:y, 0:x]
    top_right = img[0:y, x+w:img.shape[1]]
    left = img[y:y+h, 0:x]
    right = img[y:y+h, x+w:img.shape[1]]

    bg = (aveColor(top_right)+aveColor(top_left)+aveColor(left)+aveColor(right)) / 4
    #diff = img - bg
    #img[np.sum(np.absolute(diff),2) < threshold, :] = [255,255,255]
    for y in range(img.shape[0]):
    	x = 0
    	while x < img.shape[1] and isInBG(x,y,img):
    		if isBG(x,y,img):
    			img[y,x,:] = [255,255,255]
    		x += 1
    	if x < img.shape[1] - 1:
    		x = img.shape[1] - 1
    		while x >= 0 and isInBG(x,y,img):
    			if isBG(x,y,img):
    				img[y,x,:] = [255,255,255]
    			x -= 1




    #print aveColor(top_left)
    #print aveColor(left)
    #print aveColor(top_right)
    #print aveColor(right)

    #print patchDist(top_right, right)

    #f = img[y:y+h, x:x+w]
    #print patchDist(top_right, f)

#print np.mean(np.mean(img,0),0)

cv2.imshow('image', img)
cv2.waitKey(0)
cv2.destroyAllWindows()

