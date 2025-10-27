from flask import Flask,request,jsonify
import joblib
import pandas as pd
import os
from prepare_data import transform_data
app = Flask(__name__)

#โหลดโมเดลตอนเริ่ม server
Model_Path = "data/output/xgboost-model.pkl"
model = joblib.load(Model_Path)
print('Model Load Succesfully')

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        if isinstance(data,dict):
            df = pd.DataFrame([data])
        elif isinstance(data,list):
            df = pd.DataFrame(data)
        else:
            return jsonify({"error":"Invalid input format"}),400
        # แปลง feature ให้เหมือนตอน train
        df_transformed = transform_data(df)
        
        predictions = model.predict(df_transformed)
        result = predictions.tolist()
        
        return jsonify({"prediction" : result})
    
    except Exception as e:
        return jsonify({"error":str(e)}),500
    
@app.route("/",methods=["GET"])
def home():
    return jsonify({"message": "ML Serve Api is running"})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)
        
        