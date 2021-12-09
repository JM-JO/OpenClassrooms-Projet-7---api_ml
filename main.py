# 1. Library imports
import uvicorn
from fastapi import FastAPI
import joblib
import pandas as pd
import json
import shap


# 2. Create app and model objects
app = FastAPI()
model = joblib.load('./resources/API_model.joblib')
df_test_sample = joblib.load('./resources/df_test_sample.joblib')
optimum_threshold = joblib.load('./resources/optimum_threshold.joblib')


# 3A. Index route, opens automatically on http://127.0.0.1:8000
@app.get('/')
def index():
    '''
    This is a first docstring.
    '''
    return {'message': 'Hello, stranger'}


# 3B. Route with a single parameter, returns the parameter within a message
@app.get('/api/{name}')
def get_name(name: str):
    return {'message': f'Hello, {name}'}


# 4A. Retourne la data reçue
@app.post('/ping/')
def ping(incoming_data):
	print("-----------------------------------------")
	print("type :", type(incoming_data))
	print("-----------------------------------------")
	return incoming_data


# 4A2. Retourne la data reçue (autre syntaxe pour FastAPI)
@app.post('/ping2/{incoming_data}')
def ping2(incoming_data):     # il faut que l'agument de la fonction (incoming_data) soit identique à celui du décorateur
	print("-----------------------------------------")
	print("type :", type(incoming_data))
	print("-----------------------------------------")
	return incoming_data


# 4C.Prise en main de la structure GET/POST + {id_client}
@app.get('/hello_id_client/{id_client}')
@app.post('/hello_id_client/{id_client}')
def hello_id_client(id_client : int):
	print("------------------------------------------")
	print("type :", type(id_client))
	print("------------------------------------------")
	return f"Hello from hello_id_client(). Value = {id_client}. Type = {type(id_client)}"


# 4D. Convertit le json reçu en dataframe, puis retourne l'index du client
@app.post('/pong/')
def pong_json(incoming_json):
	df_one_client = pd.read_json(incoming_json, orient='index')
	var = int(df_one_client.index[0])
	return {'index client :': var}


# 5A.Convertit le json reçu (au format de dict dans un string) en dataframe, puis retourne la proba de défaut de crédit
@app.post('/predict_json/{json_one_client}')
def predict_json(json_one_client):
	df_one_client = pd.read_json(json_one_client, orient='index')
	probability = model.predict_proba(df_one_client)[:,1][0]
	return {'probability': probability}


# 5B.Convertit l'id du client, puis retourne la proba de défaut de crédit
@app.post('/predict_id_client/{id_client}')
def predict_id_client(id_client : int):
	df_one_client = df_test_sample[df_test_sample.index == id_client]
	probability = model.predict_proba(df_one_client)[:,1][0]
	return {'probability': probability}


# Retourne le optimum_threshold
@app.post('/optimum_threshold/')
def return_optimum_threshold():
    return optimum_threshold


# Returns the SHAP values (json) for a client
@app.post('/fetch_data_id_client/{id_client}')
def fetch_data_id_client(id_client : int):
	df_one_client = df_test_sample[df_test_sample.index == id_client]    # type: pandas
	return df_one_client.to_json(orient='index')


# Returns the SHAP values (json) for a client
@app.post('/fetch_shap_id_client/{id_client}')
def fetch_shap_id_client(id_client : int):
	df_one_client = df_test_sample[df_test_sample.index == id_client]    # type: pandas
	explainer = shap.TreeExplainer(model)
	shap_values = explainer.shap_values(df_one_client)
	df_shap = pd.DataFrame({'SHAP value' : shap_values[1][0], 'feature' : df_test_sample.columns})
	df_shap.sort_values(by='SHAP value', inplace=True, ascending=False)
	return df_shap.to_json(orient='index')



