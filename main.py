import op
from op.vpen_flask import *

# My Project
import Chu
from Chu.Project1 import *
from Chu.simplewebcam import *

# import Flask
from flask import Flask, render_template, Response

app = Flask(__name__)

@app.route('/')
def index():
    """Video streaming home page."""
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    """Video streaming route. Put this in the src attribute of an img tag."""
    return Response(gen(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/video_feed2')
def video_feed2():
    """Video streaming route. Put this in the src attribute of an img tag."""
    return Response(Chu(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/video_feed3',endpoint = 'video_feed3')
def video_feed3():
    """Video streaming route. Put this in the src attribute of an img tag."""
    return Response(Chu2(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

# @app.route('/video_feed4',endpoint = 'video_feed4')
# def video_feed4():
#     """Video streaming route. Put this in the src attribute of an img tag."""
#     return Response(Face(),
#                     mimetype='multipart/x-mixed-replace; boundary=frame')

# @app.route('/video_feed5',endpoint = 'video_feed5')
# def video_feed5():
#     """Video streaming route. Put this in the src attribute of an img tag."""
#     return Response(Style_transfer(),
#                     mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(host= '0.0.0.0',debug=False)    
    
