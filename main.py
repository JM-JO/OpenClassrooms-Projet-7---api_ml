# 1. Library imports
import uvicorn
from fastapi import FastAPI, Body   # Request
import joblib
import pandas as pd
import json
import shap

# 2. Create app and model objects
app = FastAPI()
model = joblib.load('./resources/API_model.joblib')
optimum_threshold = joblib.load('./resources/optimum_threshold.joblib')


# 3A. Index route, opens automatically on http://127.0.0.1:8000
@app.get('/')
def index():
    """
    This is a first docstring.
    """
    return {'message': 'Hello, stranger'}



# Retourne le optimum_threshold
@app.get('/optimum_threshold/')
def get_optimum_threshold():
    return optimum_threshold



# 5.Retourne la proba de défaut de crédit pour un client
@app.get('/prediction/')
def get_prediction(json_client: dict = Body({})):
    df_one_client = pd.Series(json_client).to_frame().transpose()
    probability = model.predict_proba(df_one_client)[:, 1][0]
    return {'probability': probability}






# Returns the SHAP values (json) for a client
@app.get('/shap/')
def get_shap(json_client: dict = Body({})):
    df_one_client = pd.Series(json_client).to_frame().transpose()
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(df_one_client)
    df_shap = pd.DataFrame({'SHAP value': shap_values[1][0], 'feature': df_one_client.columns})
    df_shap.sort_values(by='SHAP value', inplace=True, ascending=False)
    return df_shap.to_json(orient='index')
