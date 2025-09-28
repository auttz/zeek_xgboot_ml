from flask import Flask, request, jsonify

app = Flask(__name__)

# ✅ Endpoint แบบ GET
@app.route("/hello", methods=["GET"])
def hello():
    return jsonify({"message": "Hello API 🚀"})

# ✅ Endpoint แบบ POST (รับค่ามาบวกกัน)
@app.route("/add", methods=["POST"])
def add_numbers():
    data = request.get_json()  # รับ JSON จาก client
    a = data.get("a", 0)       # ดึงค่า a (ถ้าไม่มีให้เป็น 0)
    b = data.get("b", 0)       # ดึงค่า b
    return jsonify({"result": a + b})

if __name__ == "__main__":
    app.run(debug=True, port=5000)
