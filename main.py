import numpy as np
import pickle
import cv2
from os import listdir
from sklearn.preprocessing import LabelBinarizer
from keras import backend as K
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam
from keras.preprocessing import image
from keras.preprocessing.image import img_to_array
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn import preprocessing
from sklearn.externals import joblib


#K.clear_session()


labels=['Pepper__bell___Bacterial_spot','Pepper__bell___healthy',
'Potato___Early_blight','Potato___Late_blight','Potato___healthy',
'Tomato_Bacterial_spot','Tomato_Early_blight','Tomato_Late_blight',
'Tomato_Leaf_Mold','Tomato_Septoria_leaf_spot',
'Tomato_Spider_mites_Two_spotted_spider_mite','Tomato__Target_Spot',
'Tomato__Tomato_YellowLeaf__Curl_Virus','Tomato__Tomato_mosaic_virus',
'Tomato_healthy']

def convert_image_to_array(image_dir):
    try:
        image = cv2.imread(image_dir)
        if image is not None :
            image = np.reshape(image, default_image_size)   
            return img_to_array(image)
        else :
            return np.array([])
    except Exception as e:
        print("Error : e")
        return None




default_image_size = tuple((256, 256))
def getPrediction(filename):
	#file_object = 'cnn_model.pkl'
	#model=pickle.load(open(filename, 'rb'))
	print('IN MAIN.PY FILE')
	import pickle

	#with open("cnn_model300.pkl", "rb") as f:
	#	w = pickle.load(f)
	#	pickle.dump(w, open("a_py2.pkl","wb"), protocol=2)

	model = joblib.load("cnn_model300.pkl")

	#model = pickle.load("a_py2.pkl")
	#imgpath='/content/drive/My Drive/Final Project files/TEST.JPG'
	lb = preprocessing.LabelBinarizer()

	imar = convert_image_to_array(filename) 
	npimagelist = np.array([imar], dtype=np.float16)/225.0 
	PREDICTEDCLASSES2 = model.predict_classes(npimagelist) 
	num=np.asscalar(np.array([PREDICTEDCLASSES2]))
	print('process almost done')
	return labels[num]