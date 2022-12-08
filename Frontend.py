from flask import Flask, render_template, Response, request
from camera import Video
import cv2

app=Flask(__name__)

cam = cv2.VideoCapture(0)

@app.route('/')
def index():
    if request.method == 'GET':
        emotion = Video.get_emotion("test")
        recognition = Video.get_recognition("test")
        return render_template('index.html',emotion = emotion,recognition = recognition)
    else:
        return render_template('index.html')

def gen(camera):
    while True:
        frame=camera.get_frame()
        yield(b'--frame\r\n'
       b'Content-Type:  image/jpeg\r\n\r\n' + frame +
         b'\r\n\r\n')

@app.route('/video')
def video():
    return Response(gen(Video()),
    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/handleData', methods=["GET",'POST'])
def handleData():
    ret,frame = cam.read()
    Video.captureImage(frame)
    return render_template('index.html')

app.run(debug=True)