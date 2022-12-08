from flask import Flask, render_template, Response, request
from camera import Video

app=Flask(__name__)

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

app.run(debug=True)