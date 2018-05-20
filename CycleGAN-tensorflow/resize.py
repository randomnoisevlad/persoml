import glob
import cv2

files = glob.glob('test/*.jpg') + glob.glob('test/*.png')

for f in files:
	img = cv2.imread(f)
	img = cv2.resize(img, (150, 225))
	cv2.imwrite(f, img)
