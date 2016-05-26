import numpy as np
import cv2
import matplotlib.pyplot as plt

img = cv2.imread('../img/yc.jpg')
#img = cv2.imread('../img/vc.jpg')
#print img.shape

#img = cv2.resize(img,(700,500))
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

xmlPath = '/home/wuyi/Apps/opencv/data/haarcascades/'
face_cascade = cv2.CascadeClassifier(xmlPath + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(xmlPath + 'haarcascade_eye.xml')
faces = face_cascade.detectMultiScale(gray, 1.3, 3)

print len(faces)
print "********************"

for (x,y,w,h) in faces:
    cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
    roi_gray = gray[y:y+h, x:x+w]
    roi_color = img[y:y+h, x:x+w]

#    inlay = img.crop((x,y,x+w,y+h)).rotate(90)
#    img.paste(inlay,(x,y,x+w,y+h))
    
    eyes = eye_cascade.detectMultiScale(roi_gray)
    for (ex,ey,ew,eh) in eyes:
    	cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)

	


cv2.imshow('image', img)
cv2.waitKey(0)
cv2.destroyAllWindows()