import os,sys
import pandas as pd
import xgboost as xgb
import time
from sklearn.metrics import accuracy_score, classification_report
import joblib

def main(output_folder):
    # โหลดไฟล์ training / testing
    train_file = os.path.join(output_folder,'training-set.csv')
    test_file = os.path.join(output_folder,'testing-set.csv')

    if not os.path.exists(train_file) or not os.path.exists(test_file):
        sys.exit('There is no training-set or testing-set.csv')

    df_train = pd.read_csv(train_file)
    df_test = pd.read_csv(test_file)

    if 'label' not in df_train.columns:
        sys.exit('There is no label column in dataset')

    # แยก features / label
    x_train = df_train.drop(columns=['label'])
    y_train = df_train['label']
    x_test = df_test.drop(columns=['label'])
    y_test = df_test['label']
    features_used = list(x_train.columns)

    # รับ parameter จาก env var
    n_estimators = int(os.getenv("N_ESTIMATORS", 100))
    learning_rate = float(os.getenv("LEARNING_RATE", 0.1))
    max_depth = int(os.getenv("MAX_DEPTH", 6))
    random_state = int(os.getenv("RANDOM_STATE", 42))

    model = xgb.XGBClassifier(
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        max_depth=max_depth,
        random_state=random_state,
        use_label_encoder=False,
        eval_metric="logloss"
    )

    # จับเวลา train
    start_time = time.time()
    model.fit(x_train,y_train)
    end_time = time.time()
    time_duration = end_time - start_time

    params_used = {
        "n_estimators": n_estimators,
        "learning_rate": learning_rate,
        "max_depth": max_depth,
        "random_state": random_state,
        "use_label_encoder": False,
        "eval_metric": "logloss"
    }

    # predict & evaluate
    y_predict = model.predict(x_test)
    acc = accuracy_score(y_test,y_predict)
    print(f"✅ Accuracy: {acc:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_predict))

    num_features = x_train.shape[1]

    # classification report → dict → DataFrame
    report_dict = classification_report(y_test, y_predict, output_dict=True)
    df_report = pd.DataFrame(report_dict).transpose()

    # export HTML
    report_path = os.path.join(output_folder, "classification_report.html")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(f"<h2>Classification Report</h2>\n")
        f.write(f"<p>Accuracy: {acc:.4f}</p>\n")
        f.write(f"<p>จำนวนฟีเจอร์ที่ใช้ในการเทรน: {num_features}</p>\n")
        f.write(f"<p>เวลาที่ใช้ในการเทรน: {time_duration:.2f} วินาที</p>\n")

        # แสดง features
        if features_used:
            f.write("<h3>Features ที่ใช้ในการ Train</h3><ul>\n")
            for col in features_used:
                f.write(f"<li>{col}</li>\n")
            f.write("</ul>\n")
        else:
            f.write("<p><b>⚠️ ไม่มีฟีเจอร์ที่ถูกใช้งาน</b></p>\n")

        # แสดง parameters
        f.write("<h3>Parameters ที่ใช้ในการ Train xgboost model</h3>\n<ul>\n")
        for key, value in params_used.items():
            f.write(f"<li>{key}: {value}</li>\n")
        f.write("</ul>\n")

        # คั่น layout ให้ชัด
        f.write("<hr>\n")

        # ใส่ class bootstrap ให้ตารางสวยขึ้น
        f.write(df_report.to_html(classes='dataframe table table-striped'))

    print(f"📑 HTML report saved to {report_path}")

    # save model
    model_path = os.path.join(output_folder,'xgboost-model.pkl')
    joblib.dump(model,model_path)
    print(f"💾 Model saved to {model_path}")

if __name__ == '__main__':
    if len(sys.argv) != 2:
        print("วิธีใช้: python training-ml-xgboost.py <output_folder>")
        sys.exit(1)
    output_folder = sys.argv[1]
    main(output_folder)
