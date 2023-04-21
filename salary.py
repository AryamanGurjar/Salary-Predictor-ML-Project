from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)

model = joblib.load('model.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # int_features = request.form['salary']
    # print( float(x) for x in request.form.values())
    final_features = float(request.form['salary'])
    
    prediction = model.predict([[final_features]])

    output = round(prediction[0], 2)
   # print(output)
    return render_template('index.html', prediction_text='Predicted Salary: ${}'.format(output))

if __name__ == "__main__":
    app.run(debug=True)
