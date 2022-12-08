import os
import tensorflow as tf
import cv2
import numpy as np
import face_recognition


username = "Minhaz"

camera_id = 0
model = tf.keras.models.load_model("CSE499A_Model.h5")
faceDetectionPath = "haarcascade_frontalface_alt2.xml"
camera_id = camera_id
font_scale = 1.5
font = cv2.FONT_HERSHEY_PLAIN
averageEmotion = []
classNames = ["Angry", "Disgust", "Fear",
              "Happy", "Neutral", "Sad", "Surprised"]
pred = 0
webcamName = None
webcamEmotion = None


path = "ImageData"
averageFace = []
checkIfFound = []
images = []
fileNames = []
myList = os.listdir(path)
x = 1
y = 1
w = 1
h = 1

for cl in myList:
    curImg = cv2.imread(f'{path}/{cl}')
    images.append(curImg)
    fileNames.append(os.path.splitext(cl)[0])


def findEncodings(images):
    encodeList = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encodeList.append(encode)
    return encodeList

encodeListKnown = findEncodings(images)

faceDetect=cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_alt2.xml")

class Video(object):
    def __init__(self):
        self.video=cv2.VideoCapture(0)
    def __del__(self):
        self.video.release()
    def get_emotion(self):
        webcamEmotion = max(averageEmotion, key=averageEmotion.count, default=0)
        return webcamEmotion
    def get_recognition(self):
        return max(averageFace,key=averageFace.count,default=0)
    def get_username(self):
        return username
    def get_frame(self):
        face_roi = None
        ret,frame=self.video.read()
        
        face_detect = cv2.CascadeClassifier(
            cv2.data.haarcascades + faceDetectionPath)
        
        gray_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = face_detect.detectMultiScale(gray_img, 1.1, 5)
        
        for x,y,w,h in faces:
            x = x
            y = y
            w = w
            h = h
            roi_gray_img = gray_img[y:y+h, x:x+w]
            roi_color = frame[y:y+h, x:x+w]
            facess = face_detect.detectMultiScale(roi_gray_img)
            if len(facess) == 0:
                print("Face not detected")
            else:
                for (ex, ey, ew, eh) in facess:
                    face_roi = roi_color[ey: ey+eh, ex:ex + ew]
                    
        if face_roi is not None:
            final_img = cv2.resize(face_roi, (224, 224))
            final_img = np.expand_dims(final_img, axis=0)  # need 4th dimension
            final_img = final_img/255  # normalizing

            prediction = model.predict(final_img)
            pred = np.argmax(prediction[0])
            averageEmotion.append(classNames[pred])
            
        imgS = cv2.resize(frame, (0, 0), None, 0.25, 0.25)
        imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)

        facesCurFrame = face_recognition.face_locations(imgS)
        encodesCurFrame = face_recognition.face_encodings(imgS, facesCurFrame)

        for encodeFace, faceLoc in zip(encodesCurFrame, facesCurFrame):
            matches = face_recognition.compare_faces(encodeListKnown, encodeFace)
            faceDis = face_recognition.face_distance(encodeListKnown, encodeFace)
            matchIndex = np.argmin(faceDis)

            if faceDis[matchIndex] > 0.50:
                checkIfFound.append(0)

            if matches[matchIndex]:
                name = fileNames[matchIndex]
                averageFace.append(name)
                checkIfFound.append(1)

                webcamName = name
            
            
            
        ret,jpg=cv2.imencode('.jpg',frame)
        return jpg.tobytes()