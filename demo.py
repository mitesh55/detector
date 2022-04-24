import torch
from flask import Flask,render_template,Response
import cv2
import numpy as np
# pt_model = torch.hub.load('ultralytics/yolov5', 'yolov5s')
pt_model = torch.hub.load('ultralytics/yolov5', 'custom', path = 'colab_model.pt', force_reload=True)
print('done')
# pt_model = torch.load('../input/yolov5-weight/best.pt')
#'/home/mitesh/Custom_object_detector/connector C01-20220412T125447Z-001/connector C01/VID_20200226_172542.mp4'
#/home/mitesh/Custom_object_detector/connector C01-20220412T125447Z-001/connector C01/VID_20200228_122055.mp4
app=Flask(__name__)
def generate_frames():
    camera=cv2.VideoCapture(0)
    while True:
            
        ## read the camera frame
        success,frame=camera.read()
        if not success:
            break
        else:
#            print(type(frame))
            print(frame.shape)
            frame= cv2.putText(cv2.flip(frame,1),"Recording...", (0,25), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255),4)
            dim = (640,640)
            resized_1 = cv2.resize(frame, dim, interpolation = cv2.INTER_AREA)
            
            pt_model.to('cpu')
            resized = [resized_1]
            result = pt_model(resized)
            x_shape, y_shape = resized_1.shape[1], resized_1.shape[0]
            labels, cord = result.xyxyn[0][:,-1], result.xyxyn[0][:,:-1]
            n = len(labels)
            for i in range(n):
                row = cord[i]
                if row[4]>0.41:
                    x1, y1, x2, y2 = int(row[0]*x_shape), int(row[1]*y_shape), int(row[2]*x_shape), int(row[3]*y_shape)
                    bgr = (0, 255, 0)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), bgr, 2)
                    cv2.putText(frame, 'Connector', (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)
#            if c >= 0.85:
#                cv2.rectangle(frame, (l, t), (r, b), (0, 0, 255), 1)
#                cv2.putText(frame, 'Connector : {} %'.format(c*100), (l, t-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)
#                
            try:
                ret,buffer=cv2.imencode('.jpg',frame)
                frame=buffer.tobytes()

                yield(b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            except :
                pass
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video')
def video():
    return Response(generate_frames(),mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__=="__main__":
    app.run(debug=True, use_reloader=True)

