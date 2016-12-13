import numpy as np
import cv2
import os
from matplotlib import pyplot as plt
import sklearn.datasets

##### Cascade classifier
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
# face_cascade = cv2.CascadeClassifier('haarcascade_profileface.xml') # not very good, but gets a few
# face_cascade = cv2.CascadeClassifier('haarcascade_smile.xml') # Pretty good actually
# face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_alt.xml')
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml') # doesn't do well on glasses at all
# eye_cascade = cv2.CascadeClassifier('haarcascade_eye_tree_eyeglasses.xml') # Causes python to crash

def rgb2gray(rgb):

    r, g, b = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b

    return gray

def cascade_classifier(input_image,is_face,do_plot):
	# print 'input_image',input_image
	# img = cv2.imread('test.jpg')
	img = cv2.imread(input_image)
	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	# print 'gray',gray
	# print 'gray',gray.shape
	# print 'gray',type(gray[0][0])
	# http://stackoverflow.com/questions/30506126/open-cv-error-215-scn-3-scn-4-in-function-cvtcolor

	faces = face_cascade.detectMultiScale(gray, 1.3, 5)

	false_negative,false_positive = 0,0
	if (len(faces) == 0) and is_face:
		# False negative
		# print 'false negative'
		false_negative = 1
	elif (len(faces) != 0) and not is_face:
		# False positive
		# print 'false positive'
		false_positive = 1

	if do_plot:
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

	return (false_negative,false_positive)


def run_olivetti_images():
	for directory in os.listdir('att_faces'):
		if os.path.isdir('att_faces/'+directory):
			for filename in os.listdir('att_faces/'+directory):
				cascade_classifier('att_faces/'+directory+'/'+filename)

	return None

# run_olivetti_images()

def run_all_images(image_directory,do_plot):
	# Face images
	tmp1,tmp2 = 0,0
	false_negatives,false_positives = 0,0
	num_images = 0
	for directory in os.listdir(image_directory):
		if os.path.isdir(image_directory+'/'+directory):
			for filename in os.listdir(image_directory+'/'+directory):
				tmp1,tmp2 = cascade_classifier(image_directory+'/'+directory+'/'+filename,is_face=True,do_plot=do_plot)
				num_images += 1
				false_negatives += tmp1
				false_positives += tmp2

	print 'False negatives total: ', false_negatives
	print 'False positives total: ', false_positives
	print 'Total number of images: ',num_images
	return (false_negatives,false_positives,num_images)

# run_all_images('att_faces',True) # Results: 137 false negatives, 400 total images with the default
	# 166 false negatives, 400 total images with the alt
run_all_images('../wild_faces',True) # Results: 885 false negatives, 13233 total images
# run_all_images('../bioID_faces',True) # Results: 162 false negatives, 1521 total images -> not doing ppl out of frame

## Prepare flickr data set
people_list = np.loadtxt('people.txt').astype(int)
ok_dirs = []

for i in range(1,1001):
    if not i in people_list:
        ok_dirs.append(r'Flickr Images/im' + str(i) + '.jpg')

def run_all_images_flickr(faces_dir,do_plot,flickr_list):
	tmp1,tmp2 = 0,0
	false_negatives,false_positives = 0,0
	num_images = 0
	# Face images
	for directory in os.listdir(faces_dir):
		if os.path.isdir(faces_dir+'/'+directory):
			for filename in os.listdir(faces_dir+'/'+directory):
				tmp1,tmp2 = cascade_classifier(faces_dir+'/'+directory+'/'+filename,is_face=True,do_plot=do_plot)
				num_images += 1
				false_negatives += tmp1
				false_positives += tmp2

	# Non-face images
	for filepath in flickr_list:
		tmp1,tmp2 = cascade_classifier(filepath,is_face=False,do_plot=do_plot)
		num_images += 1
		false_negatives += tmp1
		false_positives += tmp2

	print 'False negatives total: ', false_negatives
	print 'False positives total: ', false_positives
	print 'Total number of images: ',num_images
	return (false_negatives,false_positives,num_images)

# run_all_images_flickr('../bioID_faces',False,ok_dirs) # False negatives total:  162, False positives total:  7, Total number of images:  2127
# (only flickr) 0 false negatives, False positives total:  7, Total number of images:  606
def run_test_images(faces_dir,do_plot,nonface_dir):
	tmp1,tmp2 = 0,0
	false_negatives,false_positives = 0,0
	num_images = 0
	# Face images
	for directory in os.listdir(faces_dir):
		if os.path.isdir(faces_dir+'/'+directory):
			for filename in os.listdir(faces_dir+'/'+directory):
				tmp1,tmp2 = cascade_classifier(faces_dir+'/'+directory+'/'+filename,is_face=True,do_plot=do_plot)
				num_images += 1
				false_negatives += tmp1
				false_positives += tmp2

	# None face images
	# for directory in os.listdir(nonface_dir):
		# if os.path.isdir(nonface_dir+'/'+directory):
	for filename in os.listdir(nonface_dir+'/'):
		tmp1,tmp2 = cascade_classifier(nonface_dir+'/'+filename,is_face=False,do_plot=do_plot)
		num_images += 1
		false_negatives += tmp1
		false_positives += tmp2

	print 'False negatives total: ', false_negatives
	print 'False positives total: ', false_positives
	print 'Total number of images: ',num_images
	return (false_negatives,false_positives,num_images)


# TODO switch false negative and false positive definitions

# run_test_images('att_faces',False,'testbackground')
# Results # False negatives total:  137, False positives total:  0, Total number of images:  400
# run_test_images('../bioID_faces',False,'testbackground')
# run_test_images('olivetti_new',False,'testbackground_new')
