import cv2
import numpy as np
import face_recognition
import os

username = input('Enter your name?\n')

path ="ImageData"
images = []
classNames = []
myList = os.listdir(path)
print(myList)

for cl in myList:
    curImg = cv2.imread(f'{path}/{cl}')
    images.append(curImg)
    classNames.append(os.path.splitext(cl)[0])
print(images)    
print(classNames)

def findEncodings(images):
    encodeList = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encodeList.append(encode)
    return encodeList

encodeListKnown = findEncodings(images)
##print(len(encodeListKnown))
##print('Encoding Complete')

cap = cv2.VideoCapture(0)

while True:
    averageFace = []
    checkIfFound = []
    success, img = cap.read()
    imgS = cv2.resize(img,(0,0),None,0.25,0.25)
    imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)
    
    facesCurFrame = face_recognition.face_locations(imgS)
    encodesCurFrame = face_recognition.face_encodings(imgS,facesCurFrame)
    
    for encodeFace,faceLoc in zip(encodesCurFrame,facesCurFrame):
        matches = face_recognition.compare_faces(encodeListKnown,encodeFace)
        faceDis = face_recognition.face_distance(encodeListKnown,encodeFace)
        ##print(faceDis)
        matchIndex = np.argmin(faceDis)
        
        if faceDis[matchIndex]> 0.50:
            checkIfFound.append(0)
            
        if matches[matchIndex]:
            name = classNames[matchIndex].upper()
            ##print(name)
            averageFace.append(name)
            checkIfFound.append(1)
            y1,x2,y2,x1 = faceLoc
            y1, x2, y2, x1 = y1*4,x2*4,y2*4,x1*4
            cv2.rectangle(img,(x1,y1),(x2,y2),(0,255,0),2)
            cv2.rectangle(img,(x1,y2-35),(x2,y2),(0,255,0),cv2.FILLED)
            cv2.putText(img,name,(x1+6,y2-6),cv2.FONT_HERSHEY_COMPLEX,1,(255,255,255),2)

    cv2.imshow('Webcam',img)
    cv2.waitKey(1)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        if max(averageFace,key=averageFace.count,default=0).lower()==username.lower() and max(checkIfFound,key=checkIfFound.count,default=0)==1:
            print("Verification successful")
            print("Welcome " + name)
        elif max(averageFace,key=averageFace.count,default=0).lower()!=username.lower() and max(checkIfFound,key=checkIfFound.count,default=0)==1:
            print("Name not found")
        else:
            print("Coud not verify!")

        break

cap.release()
cv2.destroyAllWindows()
    
