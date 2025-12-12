from flask import Flask, request, render_template
import pandas as pd
import pickle
import numpy as np
import joblib 
from tensorflow.keras.models import load_model

app = Flask(__name__)

model = load_model('customer_satisfaction_model.h5')

  
scaler = joblib.load('scaler.pkl')


pca = pickle.load(open('pca_model.pkl', 'rb'))
model2 = pickle.load(open('hybrid_pca_kmeans_model.pkl', 'rb'))

@app.route('/')
@app.route('/first') 
def first():
	return render_template('first.html')



@app.route('/home')
def home():
    return render_template('index.html')


@app.route('/predict', methods = ['POST'])
def predict():
    Gender = request.form['Gender']
    Age = request.form['Age']
    City = request.form['City']
    Membership_Type = request.form['Membership Type']
    Total_Spend = request.form['Total Spend']
    Items_Purchased = request.form['Items Purchased']
    Average_Rating_Numeric = request.form['Average Rating - Numeric']
    Discount_Applied_Boolean = request.form['Discount Applied - Boolean']
    Days_Since_Last_Purchase = request.form['Days Since Last Purchase']
    
    
    inputtestdata = [Gender,Age,City,Membership_Type,Total_Spend,Items_Purchased,Average_Rating_Numeric,Discount_Applied_Boolean,Days_Since_Last_Purchase]
    inputtestdata = [float(Gender), float(Age), float(City), float(Membership_Type), float(Total_Spend), float(Items_Purchased), float(Average_Rating_Numeric), float(Discount_Applied_Boolean), float(Days_Since_Last_Purchase)]
    
    features = np.array([inputtestdata])
    features_scaled = scaler.transform(features)

    # pred = model.predict(features_scaled)
    
    
    # model 1
    X_pca = pca.transform(features)
    cluster = model2.predict(X_pca)
    print("cluster", cluster)
    
    
    # model 2
    pred = model.predict(features_scaled)
    print(pred)
    prediction = np.argmax(pred, axis=1)
    print(prediction)
    
    
 
    if prediction[0] == 0:
        output = "Neutral"
    
    elif prediction[0] == 1:
        output = "Satisfied"
    else:
        output = "Unsatisfied"
    
    
    return render_template('index.html', gender = Gender, age=Age, city=City, 
                                        membership_type=Membership_Type, 
                                        total_spend=Total_Spend, 
                                        items_purchased=Items_Purchased, 
                                        average_rating_numeric=Average_Rating_Numeric, 
                                        discount_applied_boolean=Discount_Applied_Boolean, 
                                        days_since_last_purchase=Days_Since_Last_Purchase, 
                                        prediction_text = output)  



@app.route('/chart') 
def chart():
	return render_template('chart.html') 

@app.route('/performance') 
def performance():
	return render_template('performance.html')     

if __name__ == "__main__":
    app.run(debug=True)
