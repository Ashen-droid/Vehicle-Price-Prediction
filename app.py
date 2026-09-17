from flask import Flask, request, jsonify, render_template
import joblib
import pandas as pd

app = Flask(__name__)

model = joblib.load('car_price_predictor_model.pkl')
scaler = joblib.load('scaler.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict_price():
    try:
        data = request.json
        
        input_data = pd.DataFrame([data])
        
        scaled_data = scaler.transform(input_data)
        
        predicted_price = model.predict(scaled_data)[0]
        
        return jsonify({'estimated_price': round(predicted_price, 2)})
    
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True, port=5000)