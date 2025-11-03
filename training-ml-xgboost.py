import os, sys, time
import pandas as pd
import xgboost as xgb
from sklearn.metrics import accuracy_score, classification_report
import joblib
from jinja2 import Environment, FileSystemLoader

# ------------------------------
# 🧠 Training Pipeline
# ------------------------------
def main(output_folder):
    print("🚀 Starting XGBoost training pipeline...")
    train_file = os.path.join(output_folder, "training-set.csv")
    test_file = os.path.join(output_folder, "testing-set.csv")

    # ตรวจสอบว่าไฟล์มีอยู่จริง
    if not os.path.exists(train_file) or not os.path.exists(test_file):
        sys.exit("❌ Missing training-set.csv or testing-set.csv")

    # โหลด dataset
    df_train = pd.read_csv(train_file)
    df_test = pd.read_csv(test_file)
    print(f"📦 Loaded train: {df_train.shape}, test: {df_test.shape}")

    if "label" not in df_train.columns:
        sys.exit("❌ Missing 'label' column in dataset")

    # ------------------------------
    # แยก Features & Labels
    # ------------------------------
    X_train = df_train.drop(columns=["label"])
    y_train = df_train["label"]
    X_test = df_test.drop(columns=["label"])
    y_test = df_test["label"]

    features_used = list(X_train.columns)
    print(f"🔢 Features used: {len(features_used)}")

    # ------------------------------
    # รับ Hyperparameters จาก ENV
    # ------------------------------
    n_estimators = int(os.getenv("N_ESTIMATORS", 200))
    learning_rate = float(os.getenv("LEARNING_RATE", 0.1))
    max_depth = int(os.getenv("MAX_DEPTH", 6))
    random_state = int(os.getenv("RANDOM_STATE", 42))
    scale_pos_weight = float(os.getenv("SCALE_POS_WEIGHT", 24.0))

    params_used = {
        "n_estimators": n_estimators,
        "learning_rate": learning_rate,
        "max_depth": max_depth,
        "random_state": random_state,
        "scale_pos_weight": scale_pos_weight,
        "eval_metric": "logloss"
    }

    # ------------------------------
    # สร้างโมเดล XGBoost
    # ------------------------------
    model = xgb.XGBClassifier(
        **params_used,
        use_label_encoder=False,
        n_jobs=-1
    )

    # ------------------------------
    # ฝึกโมเดล + จับเวลา
    # ------------------------------
    print("🏋️‍♂️ Training model ...")
    start_time = time.time()
    model.fit(X_train, y_train)
    duration = time.time() - start_time
    print(f"✅ Training complete in {duration:.2f}s")

    # ------------------------------
    # ประเมินผล
    # ------------------------------
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"🎯 Accuracy: {acc * 100:.2f}%")

    report_dict = classification_report(y_test, y_pred, output_dict=True)
    for label, metrics in report_dict.items():
        if isinstance(metrics, dict):
            for key, value in metrics.items():
                if key != "support":
                    report_dict[label][key] = round(value * 100, 2)
                else:
                    report_dict[label][key] = int(value)

    df_report = pd.DataFrame(report_dict).transpose()

    # ------------------------------
    # เตรียม context สำหรับ Jinja2
    # ------------------------------
    context = {
        "accuracy": f"{acc * 100:.2f}%",
        "duration": f"{duration:.2f}s",
        "num_features": len(features_used),
        "params": params_used,
        "features": features_used,
        "report_html": df_report.to_html(
            classes="table table-striped table-bordered",
            border=0,
            float_format="%.2f"
        ),
    }

    # ------------------------------
    # render HTML report
    # ------------------------------
    print("🧾 Generating HTML report ...")
    try:
        env = Environment(loader=FileSystemLoader("templates"))
        template = env.get_template("report_template.html")
        html_output = template.render(context)
        report_path = os.path.join(output_folder, "classification_report.html")

        with open(report_path, "w", encoding="utf-8") as f:
            f.write(html_output)

        print(f"📄 HTML report saved → {report_path}")
    except Exception as e:
        print(f"⚠️ Warning: Failed to render HTML report → {e}")

    # ------------------------------
    # บันทึกโมเดล
    # ------------------------------
    model_path = os.path.join(output_folder, "xgboost-model.pkl")
    joblib.dump(model, model_path)
    print(f"💾 Model saved → {model_path}")

    print("✅ Training pipeline completed successfully.")


# ------------------------------
# เรียกใช้ผ่าน CLI
# ------------------------------
if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("วิธีใช้: python training-ml-xgboost.py <output_folder>")
        sys.exit(1)

    output_folder = sys.argv[1]
    main(output_folder)
