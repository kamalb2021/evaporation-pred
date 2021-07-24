import flask
import json
import numpy as np
import joblib
from flask import Flask, render_template, request
from tensorflow.keras.models import model_from_json

app = Flask(__name__)

@app.route("/")
def index():
	return flask.render_template('index.html')

@app.route("/predict", methods=["POST"])
def submit_file():
  if request.method == 'POST':
    json_file = open("model/model.json","r")
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    loaded_model.load_weights("model/model.h5")
    maxt = request.form.get('maxt')
    mint = request.form.get('mint')
    rh1 = request.form.get('rh1')
    rf = request.form.get('rf')
    ws = request.form.get('ws')
    X = np.array([[maxt,mint,rh1,rf,ws]])
    print(f"Params are: {maxt} {mint} {rh1} {rf} {ws}")
    scaler = joblib.load(f"model/scaler.save") 
    print("Scaler worked")
    X = scaler.transform(X)
    pred = loaded_model.predict(X)
    print("Model Loaded")
    return flask.render_template('index.html' , response = pred[0][0])

# start the flask app, allow remote connections
if __name__ == "__main__":
  # app.run(host='0.0.0.0', port=8111) # localhost 
  app.run() ##For heroku
