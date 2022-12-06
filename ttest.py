import tensorflow as tf
from tensorflow import keras
import numpy as np
import cv2
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model

import numpy as np

# Load the cascade
# funcion to detect objects and in our case the face in images using predefined face algorithm
facedetect = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')


#start cam
cap=cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)
font=cv2.FONT_HERSHEY_COMPLEX


#model = load_model("C:/Users//houssem//PycharmProjects//FR//keras_model.h5")
model= load_model("C:/Users//houssem//PycharmProjects//FR//model2//mmodel.h5")

def get_className(classNo):
	if classNo==0:
		return "gates"
	elif classNo==1:
		return "jack"
	elif classNo==2:
		return "modi"
	elif classNo==3:
		return "musk"
	elif classNo==4:
		return "trump"

while True:
	sucess, imgOrignal=cap.read()
	faces = facedetect.detectMultiScale(imgOrignal,1.3,5)
	for x,y,w,h in faces:
		crop_img=imgOrignal[y:y+h,x:x+h]
		img=cv2.resize(crop_img, (200,200))
		img=img.reshape(1, 200, 200, 3)
		prediction=model.predict(img)
		classIndex=model.predict_classes(img)
		probabilityValue=np.amax(prediction)

        #probability to determine unknown person

		probb=75.00

		if (round(probabilityValue * 100, 2) < probb):
			cv2.rectangle(imgOrignal, (x, y), (x + w, y + h), (0, 255, 0), 2)
			cv2.rectangle(imgOrignal, (x, y - 40), (x + w, y), (0, 255, 0), -2)
			cv2.putText(imgOrignal, "Unknown", (x, y - 10), font, 0.75, (255, 255, 255), 1, cv2.LINE_AA)
		elif (classIndex==0 and (round(probabilityValue * 100, 2) > probb)):
			cv2.rectangle(imgOrignal,(x,y),(x+w,y+h),(0,255,0),2)
			cv2.rectangle(imgOrignal, (x,y-40),(x+w, y), (0,255,0),-2)
			cv2.putText(imgOrignal, str(get_className(classIndex)),(x,y-10), font, 0.75, (255,255,255),1, cv2.LINE_AA)
		elif (classIndex==1 and (round(probabilityValue * 100, 2) > probb)):
			cv2.rectangle(imgOrignal,(x,y),(x+w,y+h),(0,255,0),2)
			cv2.rectangle(imgOrignal, (x,y-40),(x+w, y), (0,255,0),-2)
			cv2.putText(imgOrignal, str(get_className(classIndex)),(x,y-10), font, 0.75, (255,255,255),1, cv2.LINE_AA)


		elif (round(probabilityValue * 100, 2) > probb):

			cv2.rectangle(imgOrignal, (x, y), (x + w, y + h), (0, 255, 0), 2)
			cv2.rectangle(imgOrignal, (x, y - 40), (x + w, y), (0, 255, 0), -2)
			cv2.putText(imgOrignal, str(get_className(classIndex)), (x, y - 10), font, 0.75, (255, 255, 255), 1,cv2.LINE_AA)
		cv2.putText(imgOrignal,str(round(probabilityValue*100, 2))+"%" ,(180, 75), font, 0.75, (255,0,0),2, cv2.LINE_AA)
	cv2.imshow("Result",imgOrignal)
	k=cv2.waitKey(1)
	if k==ord('q'):
		break


cap.release()
cv2.destroyAllWindows()