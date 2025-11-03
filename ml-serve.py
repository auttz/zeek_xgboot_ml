from flask import Flask, request, jsonify
import joblib
import pandas as pd
import os
from prepare_data import transform_data

app = Flask(__name__)

# -----------------------------
# ✅ โหลดโมเดลตอนเริ่ม server
# -----------------------------
MODEL_PATH = "data/output/xgboost-model.pkl"
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"❌ Model not found: {MODEL_PATH}")

model = joblib.load(MODEL_PATH)
print("✅ Model loaded successfully.")


# -----------------------------
# 🔮 พยากรณ์ผ่าน API
# -----------------------------
@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()

        # รองรับทั้ง JSON object และ list
        if isinstance(data, dict):
            df = pd.DataFrame([data])
        elif isinstance(data, list):
            df = pd.DataFrame(data)
        else:
            return jsonify({"error": "Invalid input format (must be JSON object or array)"}), 400

        # 🧠 แปลงฟีเจอร์ให้เหมือนตอนเทรน
        df_transformed = transform_data(df)

        # ✅ ลบ label ออกถ้ามี (กัน feature mismatch)
        if "label" in df_transformed.columns:
            df_transformed = df_transformed.drop(columns=["label"])

        # 🧩 DEBUG LOG: ตรวจว่าข้อมูลส่งเข้าโมเดลเป็นอะไร
        print("\n🧠 [DEBUG] Features passed to model:")
        print(list(df_transformed.columns))
        print("\n🧩 [DEBUG] Sample transformed row:")
        print(df_transformed.head(1).to_dict(orient="records"))

        # 🔮 Predict
        predictions = model.predict(df_transformed)
        result = predictions.tolist()

        # 🧾 แปลงเป็นข้อความอ่านง่าย
        label_map = {0: "Normal", 1: "Malicious"}
        readable_results = [label_map.get(pred, "Unknown") for pred in result]

        # 🔁 ถ้ามีแค่ 1 record ให้ตอบเป็น string เดียว
        if len(readable_results) == 1:
            readable_results = readable_results[0]

        return jsonify({
            "prediction": result,
            "label": readable_results
        })

    except Exception as e:
        print("❌ [ERROR]", str(e))
        return jsonify({"error": str(e)}), 500


# -----------------------------
# 🏠 Health Check
# -----------------------------
@app.route("/", methods=["GET"])
def home():
    return jsonify({"message": "🚀 ML Serve API is running"})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)
