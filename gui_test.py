import PySimpleGUI as sg
import os.path
import tensorflow as tf
from tensorflow import keras
import numpy as np
import cv2
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
import numpy as np


# this function represents are our images classes

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


# this funtion is responsible for detecting the face from cam
def testcam():
 facedetect = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

 cap = cv2.VideoCapture(0)
 cap.set(3, 640)
 cap.set(4, 480)
 font = cv2.FONT_HERSHEY_COMPLEX

 model = load_model("C:/Users//houssem//PycharmProjects//FR//keras_model.h5")
    # model= load_model("C:/Users//houssem//PycharmProjects//FR//model2//mmodel.h5")
 while True:
    sucess, imgOrignal = cap.read()
    faces = facedetect.detectMultiScale(imgOrignal, 1.3, 5)
    for x, y, w, h in faces:
        crop_img = imgOrignal[y:y + h, x:x + h]
        img = cv2.resize(crop_img, (224, 224))
        img = img.reshape(1, 224, 224, 3)
        prediction = model.predict(img)
        classIndex = model.predict_classes(img)
        probabilityValue = np.amax(prediction)
        if (round(probabilityValue * 100, 2)<60.00):
            cv2.rectangle(imgOrignal, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.rectangle(imgOrignal, (x, y - 40), (x + w, y), (0, 255, 0), -2)
            cv2.putText(imgOrignal, "Unknown", (x, y - 10), font, 0.75,(255, 255, 255), 1, cv2.LINE_AA)

        if classIndex == 0:
            cv2.rectangle(imgOrignal, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.rectangle(imgOrignal, (x, y - 40), (x + w, y), (0, 255, 0), -2)
            cv2.putText(imgOrignal, str(get_className(classIndex)), (x, y - 10), font, 0.75,
                        (255, 255, 255), 1, cv2.LINE_AA)
        elif classIndex == 1:
            cv2.rectangle(imgOrignal, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.rectangle(imgOrignal, (x, y - 40), (x + w, y), (0, 255, 0), -2)
            cv2.putText(imgOrignal, str(get_className(classIndex)), (x, y - 10), font, 0.75,
                        (255, 255, 255), 1, cv2.LINE_AA)

        cv2.putText(imgOrignal, str(round(probabilityValue * 100, 2)) + "%", (180, 75), font, 0.75,
                    (255, 0, 0), 2, cv2.LINE_AA)
    cv2.imshow("Result", imgOrignal)
    k = cv2.waitKey(1)
    if k == ord('q'):
        break
 sg.Text("the result is: ", imgOrignal, str(get_className(classIndex)), (x, y - 10), font, 0.75,
        (255, 255, 255), 1, cv2.LINE_AA)

 cap.release()
 cv2.destroyAllWindows()



#this function is responsible for detecting face of an external image
def testimage(path):

 model= load_model("C:/Users//houssem//PycharmProjects//FR//model2//mmodel.h5")
 #model = load_model("C:/Users//houssem//PycharmProjects//FR//model2//_my_model.h5")


# Load the cascade
 face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

 font=cv2.FONT_HERSHEY_COMPLEX
#C:/Users//houssem//PycharmProjects//FR//images//gates//gates0.jpg
# Read the input image

 img = cv2.imread(path)
 img = cv2.resize(img,(200,200),3)
 vector_img = np.array(img, dtype=np.float16)

 print("vector shape",vector_img.shape)
 print("vector",vector_img)
 gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#classIndexx = np.argmax(model.predict(gray), axis=-1)
#print(classIndexx)
#img =img.reshape(1, 224, 224, 3)
# Convert into grayscale
# Detect faces

 faces = face_cascade.detectMultiScale(img, 1.1, 4)

# Draw rectangle around the faces
 for (x, y, w, h) in faces:
    crop_img = img[y:y + h, x:x + h]
    imgg = cv2.resize(crop_img, (200, 200),3)
    imgg = img.reshape(1, 200, 200, 3)

    prediction = model.predict(imgg)
    print(prediction)
    classIndex = np.argmax(model.predict(imgg), axis=-1)

    probabilityValue = np.amax(prediction)

    print(classIndex)

    imgOrignal=img

    # probability to determine unknown person
    prob=75.00
    if (round(probabilityValue * 100, 2) < prob):
        cv2.rectangle(imgOrignal, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.rectangle(imgOrignal, (x, y - 40), (x + w, y), (0, 255, 0), -2)
        cv2.putText(imgOrignal, "Unknown", (x, y - 10), font, 0.75, (255, 255, 255), 1, cv2.LINE_AA)

    elif   (classIndex == 0 and (round(probabilityValue * 100, 2) > prob)):
         cv2.rectangle(imgOrignal, (x, y), (x + w, y + h), (0, 255, 0), 2)
         cv2.rectangle(imgOrignal, (x, y - 40), (x + w, y), (0, 255, 0), -2)
         cv2.putText(imgOrignal, str(get_className(classIndex)), (x, y - 10), font, 0.75, (255, 255, 255), 1, cv2.LINE_AA)
    elif (classIndex == 1 and (round(probabilityValue * 100, 2) > prob)):
         cv2.rectangle(imgOrignal, (x, y), (x + w, y + h), (0, 255, 0), 2)
         cv2.rectangle(imgOrignal, (x, y - 40), (x + w, y), (0, 255, 0), -2)
         cv2.putText(imgOrignal, str(get_className(classIndex)), (x, y - 10), font, 0.75, (255, 255, 255), 1, cv2.LINE_AA)
    elif (round(probabilityValue * 100, 2) > prob):
	     cv2.rectangle(imgOrignal, (x, y), (x + w, y + h), (0, 255, 0), 2)
	     cv2.rectangle(imgOrignal, (x, y - 40), (x + w, y), (0, 255, 0), -2)
	     cv2.putText(imgOrignal, str(get_className(classIndex)), (x, y - 10), font, 0.75, (255, 255, 255), 1, cv2.LINE_AA)

# if classIndex == 0:
# 	cv2.rectangle(imgOrignal, (x, y), (x + w, y + h), (0, 255, 0), 2)
# 	cv2.rectangle(imgOrignal, (x, y - 40), (x + w, y), (0, 255, 0), -2)
# 	cv2.putText(imgOrignal, str(get_className(classIndex)), (x, y - 10), font, 0.75, (255, 255, 255), 1, cv2.LINE_AA)
# elif classIndex == 1:
# 	cv2.rectangle(imgOrignal, (x, y), (x + w, y + h), (0, 255, 0), 2)
# 	cv2.rectangle(imgOrignal, (x, y - 40), (x + w, y), (0, 255, 0), -2)
# 	cv2.putText(imgOrignal, str(get_className(classIndex)), (x, y - 10), font, 0.75, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(imgOrignal, str(round(probabilityValue * 100, 2)) + "%", (80, 175), font, 0.75, (255, 0, 0), 2, cv2.LINE_AA)

# cv2.rectangle(gray, (x, y), (x + w, y + h), (0, 255, 0), 2)
    # cv2.rectangle(gray, (x, y - 40), (x + w, y), (0, 255, 0), -2)
    # #cv2.putText(gray, str(get_className(classIndex)), (x, y - 10), font, 0.75, (255, 255, 255), 1, cv2.LINE_AA)
    # # prediction = model.predict(img)
    # # classIndex = model.predict_classes(img)
    # # probabilityValue = np.amax(prediction)
    # cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)

# Display the output
 cv2.imshow('image', img)
 cv2.waitKey()




# The GUI CODE

file_list_column = [
    [
    sg.Text("Image Folder"),
    sg.In(size=(25,1), enable_events=True, key="-FOLDER-"),
    sg.FolderBrowse(),
   ],
   [
     sg.Listbox(
         values=[], enable_events=True, size= (40,20),key="-FILE LIST-"
     )
   ],

]

image_viewer_column=[
    [sg.Text("Choose an image from the list: ")],
    [sg.Button("start cam")],
    [sg.Button("recognize image")],

    [sg.Button("Collect images")],
    [sg.Button("Start CNN")],
    [sg.Text("Training accuracy: ")],
    [sg.Text(size=(40, 1), key="-trainR-")],
    [sg.Text("Validation accuracy: ")],
    [sg.Text(size=(40, 1), key="-validationR-")],
    [sg.Text(size=(40,1), key="-TOUT-")],
    [sg.Image(key="-IMAGE-")],
]

layout = [
    [
    sg.Column(file_list_column),
    sg.VSeparator(),
    sg.Column(image_viewer_column),
        ]
]
window = sg.Window("FACIAL RECOGNITION", layout)




while True:
    event, values = window.read()
    if event=="start cam":
        try:
            os.system('python ttest.py')

        except:

            pass

    if event=="Start CNN":
        try:
            import building_net

            print("---------------------- training completed -------------------------")
            # train_model()
            # os.system('python rgb_import.py')
            acct1 = building_net.acc_train1
            accv1 = building_net.acc_val1
            # print(acct[-1], accv[-1])
            t = window["-trainR-"].update(acct1[-1])
            v = window["-validationR-"].update(accv1[-1])

        except:
            pass

    if event=="Collect images":
        try:
            os.system('python collect.py')

        except:
            pass
    if event=="recognize image":
        try:

            filename = os.path.join(
                values["-FOLDER-"], values["-FILE LIST-"][0]

            )
            print(filename)
            pathimg = filename
            testimage(pathimg)

        except:
            pass
    if event == "Exit" or event== sg.WIN_CLOSED:
        break
    if event=="-FOLDER-":
        folder = values["-FOLDER-"]
        try:
            file_list = os.listdir(folder)
        except:
            file_list=[]
        fnames = [
            f
          for f in file_list
          if os.path.isfile(os.path.join(folder, f))
          and f.lower().endswith((".png", ".gif"))

        ]
        window["-FILE LIST-"].update(fnames)
    elif event== "-FILE LIST-":
        try:
            filename= os.path.join(
                values["-FOLDER-"], values["-FILE LIST-"][0]

            )
            print(filename)
            pathimg= filename
            #testimage(pathimg)
            pt=window["-TOUT-"].update(filename)

            window["-IMAGE-"].update(filename=filename)



        except:
            pass








