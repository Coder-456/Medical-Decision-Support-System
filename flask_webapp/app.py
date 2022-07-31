from flask import Flask, render_template, request,jsonify
from flask_ngrok import run_with_ngrok
from keras.models import load_model
from flask_debugtoolbar import DebugToolbarExtension

import cv2
import numpy as np
import base64
from PIL import Image
import io
import re

img_size=100

app = Flask(__name__, template_folder='/content/drive/MyDrive/flask_webapp/templates',static_folder='/content/drive/MyDrive/flask_webapp/static') 
run_with_ngrok(app)
app.debug = True

model=load_model('/content/drive/MyDrive/model-016.model')

label_dict={0:'Covid19 Negative', 1:'Covid19 Positive'}

def preprocess(img):

	img=np.array(img)

	if(img.ndim==3):
		gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
	else:
		gray=img

	gray=gray/255
	resized=cv2.resize(gray,(img_size,img_size))
	reshaped=resized.reshape(1,img_size,img_size)
	return reshaped

@app.route("/")
def index():
	return(render_template("index.html"))

@app.route("/watiscorona")
def watiscorona():
	return(render_template("watiscorona.html"))

@app.route("/abtproj")
def abtproj():
	return(render_template("abtproj.html"))

@app.route("/team")
def team():
	return(render_template("team.html"))
 
@app.route("/predict", methods=["POST"])
def predict():
	message = request.get_json(force=True)
	encoded = message['image']
	decoded = base64.b64decode(encoded)
	dataBytesIO=io.BytesIO(decoded)
	dataBytesIO.seek(0)
	image = Image.open(dataBytesIO)

	test_image=preprocess(image)

	prediction = model.predict(test_image)
	result=np.argmax(prediction,axis=1)[0]
	accuracy=float(np.argmax(prediction,axis=1)[0])

	label=label_dict[result]

	print(prediction,result,accuracy)

	response = {'prediction': {'result': label,'accuracy': accuracy}}

	return jsonify(response)
 
app.run()
