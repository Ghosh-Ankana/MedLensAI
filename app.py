
from flask import Flask, request, render_template

import json , joblib

import numpy as np

import xgboost as xgb

import shap



app = Flask(__name__)

model = xgb.XGBClassifier()
model.load_model('model.json')
explainer = shap.TreeExplainer(model)

def preprocess(symptom_labels, user_symptoms):
	user_data = np.zeros(377, dtype=np.float64)

	for x in user_symptoms:
		index = symptom_labels.index(x)
		user_data[index] = 1.0

	#symptom_labels has symptom names
	#user_symptoms is the checked symptoms
	#user_data is the list of 377 0s 
	return np.array([user_data])

def makePrediction(symptom_labels, user_data, user_input, label_encoder):
	probabilities = model.predict_proba(user_data)
	max_prob_index = np.argmax(probabilities, axis=1)
	predicted_label = label_encoder.inverse_transform(max_prob_index)[0]
	confidence = np.max(probabilities) * 100
	shap_values = explainer.shap_values(user_data)[0][0]

	si = np.argsort(probabilities[0])
	second_prob_index = si[-2]
	predicted_label2 = label_encoder.inverse_transform([second_prob_index])[0]
	confidence2 = probabilities[0, second_prob_index] * 100
	all_shap = explainer.shap_values(user_data)
	shap_values2 = all_shap[0, :, second_prob_index]

	least_index = si[0]
	predicted_label3 = label_encoder.inverse_transform([least_index])[0]
	confidence3 = probabilities[0, least_index] * 100

	
	print(shap_values)
	contributing_pos_symptoms = []
	contributing_neg_symptoms = []
	for x in range(5):
		max_sv_index = np.argmax(shap_values)
		symptom = symptom_labels[max_sv_index]
		if symptom in user_input:
			contributing_pos_symptoms.append(symptom)
		else:
			contributing_neg_symptoms.append(symptom)
		shap_values[max_sv_index] = -np.inf
	
	contributing_pos_symptoms2 = []
	contributing_neg_symptoms2 = []
	for x in range(5):
		max_sv_index = np.argmax(shap_values2)
		symptom2 = symptom_labels[max_sv_index]
		if symptom2 in user_input:
			contributing_pos_symptoms2.append(symptom2)
		else:
			contributing_neg_symptoms2.append(symptom2)
		shap_values2[max_sv_index] = -np.inf


	return {'Most LikelyPrediction': (predicted_label, confidence), 
	'Contributing Positive Symptoms': contributing_pos_symptoms, 
	'CNS': contributing_neg_symptoms,
	'second prediction': (predicted_label2, confidence2),
	'least prediction': (predicted_label3, confidence3),
	'CPS2': contributing_pos_symptoms2,
	'CNS2': contributing_neg_symptoms2}


@app.route('/', methods=['GET', 'POST'])
def home():
	results = None

	with open("symptoms.json") as f:
		symptoms = json.load(f)

	label_encoder = joblib.load("label_encoder.pkl")

	if request.method == 'POST':
		user_input = request.form.getlist('checked')
		user_data = preprocess(symptoms, user_input)
		results = makePrediction(symptoms, user_data, user_input, label_encoder)
		print(results)

	return render_template('index.html', results=results)

@app.route('/health_knowledge')
def health_knowledge():
	return render_template('health_knowledge.html')

@app.route('/about_model')
def about_model():
	return render_template('about_model.html')

@app.route('/find_care')
def find_care():
	return render_template('find_care.html')

@app.route('/about_me')
def about_me():
	return render_template('about_me.html')

if __name__ == '__main__':
	app.run()
