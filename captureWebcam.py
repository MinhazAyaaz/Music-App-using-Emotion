import cv2
import os

name = input('Enter your name?\n')

path = 'E:/Music-App-using-Emotion/ImageData'

cam = cv2.VideoCapture(0)

img_counter = 0

while True:
    ret,frame = cam.read()
    
    if not ret:
        print("Failed to grab frame")
        break
    
    cv2.imshow("test",frame)
    
    k = cv2.waitKey(1)
    
    if k%256 == 27:
        print("Escape hit, closing the app")
        break
    
    elif k%256 == 32:
        img_name = "{}.png".format(name)
        
        cv2.imwrite(os.path.join(path ,img_name),frame)
        
        print("Sign In Successfull!")
        img_counter += 1
        
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


cam.release()
cv2.destroyAllWindows()