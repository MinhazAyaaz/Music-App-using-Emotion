import os
import tensorflow as tf
import cv2
import numpy as np
import face_recognition

# username = input('Enter your name?\n')
username = "Minhaz"

model = tf.keras.models.load_model("CSE499A_Model.h5")

camera_id = 0
faceDetectionPath = "haarcascade_frontalface_alt2.xml"
camera_id = camera_id
font_scale = 1.5
font = cv2.FONT_HERSHEY_PLAIN
averageEmotion = []
classNames = ["Angry", "Disgust", "Fear",
              "Happy", "Neutral", "Sad", "Surprised"]
pred = None
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


cap = cv2.VideoCapture(camera_id)

# Check if the webcam is open correctly
if not cap.isOpened():
    raise IOError("Can't open Webcam")

while True:
    face_roi = None
    ret, frame = cap.read()

    face_detect = cv2.CascadeClassifier(
        cv2.data.haarcascades + faceDetectionPath)

    gray_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_detect.detectMultiScale(gray_img, 1.1, 5)

    for x, y, w, h in faces:
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
            webcamName = "unknown"

        if matches[matchIndex]:
            name = fileNames[matchIndex]
            averageFace.append(name)
            checkIfFound.append(1)

            webcamName = name
    
    webcamEmotion = classNames[pred]
    
    cv2.rectangle(frame, (x, y), (x+w, y+h), (255,255,0),5)
    
    cv2.rectangle(frame, (0,0), (150,50),(255,255,0),cv2.FILLED)
    cv2.putText(frame, webcamName, (10,30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8,(0,0,0), 2)
    
    cv2.rectangle(frame, (0,60), (150,110),(255,0,0),cv2.FILLED)
    cv2.putText(frame, webcamEmotion, (10,90),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8,(255,255,255), 2)
    
    cv2.imshow("Face emotion recognition", frame)
    
    

    if cv2.waitKey(1) & 0xFF == ord("q"):
        if max(averageFace,key=averageFace.count,default=0).lower()==username.lower() and max(checkIfFound,key=checkIfFound.count,default=0)==1:
            print("Verification successful")
            print("Welcome " + name)
            print("Emotion Detected:")
            print(max(averageEmotion, key=averageEmotion.count, default=0))
        elif max(averageFace,key=averageFace.count,default=0).lower()!=username.lower() and max(checkIfFound,key=checkIfFound.count,default=0)==1:
            print("Name not found")
        else:
            print("Coud not verify!")
        break

cap.release()
cv2.destroyAllWindows()