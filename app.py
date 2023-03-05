import pickle
from flask import Flask, app,jsonify,request,render_template, url_for
import pandas as pd
import numpy as np

app = Flask(__name__)

regmodel = pickle.load(open('regmodel.pkl','rb'))
scalar=pickle.load(open('scaling.pkl','rb'))
@app.route('/')
def home():
    return render_template("home.html")

@app.route('/predict_api',methods=['POST'])
def predict_api():
    data = request.json['data']
    new_data = scalar.transform(np.array(list(data.values())).reshape(1,-1))
    output=regmodel.predict(new_data)
    return jsonify(output[0])

@app.route('/predict',methods=['POST'])
def predict():
    data=[float(x) for x in request.form.values()]
    new_data=scalar.transform((np.array(data)).reshape(1,-1))
    output=regmodel.predict(new_data)[0]
    return render_template("home.html",prediction_text="Predicted House Price is {}".format(output))

if __name__ == "__main__":
    app.run(debug = True)