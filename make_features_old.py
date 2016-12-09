import numpy as np
import integral_image_transform
import sklearn.datasets

def get_faces():
	return sklearn.datasets.fetch_olivetti_faces()

images = get_faces()['images']
labels = get_faces()['labels']

def compute_two_rectangle_features():