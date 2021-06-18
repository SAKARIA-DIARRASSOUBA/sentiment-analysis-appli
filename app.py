from flask import Flask, render_template,request
import pandas as pd
import numpy as numpy
import joblib 
#import pickle
#from tensorflow.keras.models import load_model

with open('Classifier_rf','rb') as file:

    model=joblib.load(file)

#model=load_model('model')

with open("tfidfmodel.pickle","rb") as f:
	tfidf=joblib.load(f)
#tfidf=pickle.load(open("tfidfmodel.pickle","rb"))
#model=joblib.load("model")

app = Flask(__name__)

@app.route('/', methods=['GET'])
def home():
	return render_template("home.html")




@app.route('/predict', methods=['POST'])
def predict(): 
	#req_data = request.form(force=True)
	comment1=request.form["comment"]
	comment=[comment1]
	processing=tfidf.transform(comment).toarray()
	prediction=model.predict(processing)[:,1]
	print(" modele prediction :", prediction)

	sentiment="inconnu"

	if prediction[0] > 0.5:
		sentiment="Positive 🙂🙂🙂🙂"
	else:
		sentiment="Negative 🙁🙁🙁🙁"

	return render_template("home.html", comment=comment1,sentiment =sentiment)


if __name__ == '__main__':
    app.run(debug=True, port=8000)

