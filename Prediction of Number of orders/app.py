# -*- coding: utf-8 -*-
"""
Éditeur de Spyder

Ceci est un script temporaire.
"""
from flask import Flask, request, jsonify, render_template
import pickle
import pandas as pd
import json
import numpy as np 
from MyModel import My_Model
app=Flask(__name__)

model=pickle.load(open('Model_V_2.sav','rb'))

info=pd.read_csv('fulfilment_center_info.csv')
meal_info=pd.read_csv('meal_info.csv')
@app.route("/",methods=['GET'])
def home():
    centers=list(info['center_id'].unique())
    meals=list(meal_info['meal_id'].unique())
    return render_template('front_1.html',centers=centers,meals=meals)



    
    


@app.route("/predict",methods=['POST'])

def predict():
    centers=list(info['center_id'].unique())
    meals=list(meal_info['meal_id'].unique())
    A={'week':float(request.form['week']),'center_id':float(request.form['center_id']),'meal_id':float(request.form['meal_id']),'checkout_price':float(request.form['checkout_price']),'base_price':float(request.form['base_price']),'emailer_for_promotion':float(request.form['emailer_for_promotion']),'homepage_featured':float(request.form['homepage_featured'])}
    df=pd.DataFrame(A,index=[0])
    Y_predicted=model.predict(df)
    return render_template('front_1.html',centers=centers,meals=meals,prediction_text='The amount predicted  {} '.format(np.round(Y_predicted[0]),2))
    

if __name__=='__main__':
    app.run(debug=True)
    
    