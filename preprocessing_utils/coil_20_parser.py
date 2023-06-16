from chainer.dataset import download as dl
import zipfile
import numpy as np
import tempfile, os, shutil, cv2, re

def imreadBGR(path):
	'''use opencv to load jpeg, png image as numpy arrays, the speed is triple compared with skimage
	'''
	return cv2.imread(path, 3)


def download(dataset_type='unprocessed'):
	if dataset_type=='unprocessed':
		url = "http://www.cs.columbia.edu/CAVE/databases/SLAM_coil-20_coil-100/coil-20/coil-20-unproc.zip"
	elif dataset_type=='processed':
		url = 'http://www.cs.columbia.edu/CAVE/databases/SLAM_coil-20_coil-100/coil-20/coil-20-proc.zip'
	else:
		raise ValueError("dataset_type should be either unprocessed or processed")

	archive_path = dl.cached_download(url)
	fileOb = zipfile.ZipFile(archive_path, mode='r')
	names = fileOb.namelist()

	cache_root = "./temp/"
	try:
		os.makedirs(cache_root)
	except OSError:
		if not os.path.isdir(cache_root):
			raise
	cache_path = tempfile.mkdtemp(dir=cache_root)

	data, label = [], []

	try:
		for name in names:
				path = cache_path+name
				if bool(re.search('obj', name)):
					img = imreadBGR(fileOb.extract(name, path=path))
					data.append(img)
					label.append(int(name.split("__")[0].split("/obj")[1]))
	finally:
		shutil.rmtree(cache_root)


	data = np.stack(data, axis=0).transpose([0, 3, 1, 2])
	label = np.array(label).astype(np.uint8)
	data = np.reshape(data, (data.shape[0], -1))
	return data, label