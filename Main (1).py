'''
detect and segement potential nuclei in miscropic images (H&E stained)
@author: Kemeng Chen 
'''
import os
import numpy as np 
import cv2
from time import time
from util import*
import matplotlib.pyplot as plt
from skimage.feature import peak_local_max
from skimage.morphology import watershed
from scipy import ndimage
import imutils


def process(data_folder, model_name, format):
	patch_size=128
	stride=16
	file_path=os.path.join(os.getcwd(), data_folder)
	name_list=os.listdir(file_path)
	print(str(len(name_list)), ' files detected')
	model_path=os.path.join(os.getcwd(), 'models')
	model=restored_model(os.path.join(model_path, model_name), model_path)
	print('Start time:')
	print_ctime()
	

	for index, temp_name in enumerate(name_list):
		ts=time()
		print('process: ', str(index), ' name: ', temp_name)
		temp_path=os.path.join(file_path, temp_name)
		if not os.path.isdir(temp_path):
			continue
		# result_path=os.path.join(temp_path, 'mask.png')
		print(temp_name+format)
		temp_image=cv2.imread(os.path.join(temp_path, temp_name+format))
		if temp_image is None:
			raise AssertionError(temp_path, ' not found')
		#count(temp_image,temp_name+format)
		batch_group, shape=preprocess(temp_image, patch_size, stride, temp_path)
		mask_list=sess_interference(model, batch_group)
		print("count---------------------------")
		
		c_mask=patch2image(mask_list, patch_size, stride, shape)
		c_mask=cv2.medianBlur((255*c_mask).astype(np.uint8), 3)
		c_mask=c_mask.astype(np.float)/255
		#print(c_mask)
		thr=0.5
		c_mask[c_mask<thr]=0
		c_mask[c_mask>=thr]=1
		center_edge_mask, gray_map=center_edge(c_mask, temp_image)
		#print(center_edge_mask)
		print(gray_map)
		cv2.imwrite(os.path.join(temp_path, 'mask.png'), gray_map)
		cv2.imwrite(os.path.join(temp_path, 'label.png'), center_edge_mask)
		te=time()
		print('Time cost: ', str(te-ts))
		fig, ax=plt.subplots(1,2)
		ax[0].imshow(cv2.cvtColor(center_edge_mask, cv2.COLOR_BGR2RGB))
		ax[0].set_title('label')
		ax[1].imshow(gray_map)
		ax[1].set_title('Center and contour')
	
		
	model.close_sess()
	print('mask generation done')
	print_ctime()
	plt.show()

def count(img,image):
	#print(img)
	image = cv2.imread("sample_3.png")
	#temp_image=cv2.imread(os.path.join(temp_path, temp_name+format))
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	thresh = cv2.threshold(gray, 0, 255,
    cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
	cv2.imshow("Thresh", thresh)


	D = ndimage.distance_transform_edt(thresh)
	localMax = peak_local_max(D, indices=False, min_distance=20,
    labels=thresh)
	cv2.imshow("D image", D)

	markers = ndimage.label(localMax, structure=np.ones((3, 3)))[0]
	labels = watershed(-D, markers, mask=thresh)
	print("[INFO] {} unique segments found".format(len(np.unique(labels)) -     1))

	for label in np.unique(labels):
		if label == 0:
			continue

		mask = np.zeros(gray.shape, dtype="uint8")
		mask[labels == label] = 255
		# detect contours in the mask and grab the largest one
		cnts = cv2.findContours(mask.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
		cnts = imutils.grab_contours(cnts)
		c = max(cnts, key=cv2.contourArea)
		# draw a circle enclosing the object
		((x, y), r) = cv2.minEnclosingCircle(c)
		cv2.circle(image, (int(x), int(y)), int(r), (0, 255, 0), 2)
		cv2.putText(image, "#{}".format(label), (int(x) - 10, int(y)),cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
	cv2.imshow("input",image)
	cv2.waitKey(0)

def main():
	data_folder='data'
	model_name='nucles_model_v3.meta'
	format='.png'
	process(data_folder, model_name, format)

if __name__ == '__main__':
	main()
