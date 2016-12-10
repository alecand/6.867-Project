import numpy as np
import cv2
import os
from matplotlib import pyplot as plt
import sklearn.datasets

##### Cascade classifier
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
# face_cascade = cv2.CascadeClassifier('haarcascade_profileface.xml') # not very good, but gets a few
# face_cascade = cv2.CascadeClassifier('haarcascade_smile.xml') # Pretty good actually
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')

def cascade_classifier(input_image):
	print 'input_image',input_image
	# img = cv2.imread('test.jpg')
	img = cv2.imread(input_image)
	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	# http://stackoverflow.com/questions/30506126/open-cv-error-215-scn-3-scn-4-in-function-cvtcolor

	faces = face_cascade.detectMultiScale(gray, 1.3, 5)
	for (x,y,w,h) in faces:
		cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
		roi_gray = gray[y:y+h, x:x+w]
		roi_color = img[y:y+h, x:x+w]
		eyes = eye_cascade.detectMultiScale(roi_gray)
		for (ex,ey,ew,eh) in eyes:
			cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)

	# cv2.imshow('img',img)
	# cv2.waitKey(0)
	# cv2.destroyAllWindows()
	plt.imshow(img)
	plt.show()

	return None


### Import Olivetti dataset
# def get_faces():
# 	return sklearn.datasets.fetch_olivetti_faces()

# images = get_faces()['images']
# # labels = get_faces()['target']
# test_image = images[0]

def run_olivetti_images():
	for directory in os.listdir('att_faces'):
		if os.path.isdir('att_faces/'+directory):
			for filename in os.listdir('att_faces/'+directory):
				cascade_classifier('att_faces/'+directory+'/'+filename)

	return None

run_olivetti_images()