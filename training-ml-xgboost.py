import os, sys
import pandas as pd
import xgboost as xgb
import time
from sklearn.metrics import accuracy_score, classification_report
import joblib
from jinja2 import Environment, FileSystemLoader

def main(output_folder):
    # โหลดไฟล์ training / testing
    train_file = os.path.join(output_folder, 'training-set.csv')
    test_file = os.path.join(output_folder, 'testing-set.csv')

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
    scale_pos_weight = float(os.getenv("SCALE_POS_WEIGHT", 24))


    model = xgb.XGBClassifier(
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        max_depth=max_depth,
        random_state=random_state,
        use_label_encoder=False,
        eval_metric="logloss",
        scale_pos_weight=scale_pos_weight
    )

    # จับเวลา train
    start_time = time.time()
    model.fit(x_train, y_train)
    end_time = time.time()
    time_duration = end_time - start_time

    params_used = {
        "n_estimators": n_estimators,
        "learning_rate": learning_rate,
        "max_depth": max_depth,
        "random_state": random_state,
        "use_label_encoder": False,
        "eval_metric": "logloss",
        "scale_pos_weight": scale_pos_weight    
    }

    # predict & evaluate
    y_predict = model.predict(x_test)
    acc = accuracy_score(y_test, y_predict)
    print(f"✅ Accuracy: {acc * 100:.2f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_predict))
    num_features = x_train.shape[1]

    # classification report → dict → แปลงค่าเป็นเปอร์เซ็นต์ → DataFrame
    report_dict = classification_report(y_test, y_predict, output_dict=True)
    for label, metrics in report_dict.items():
        if isinstance(metrics, dict):
            for key, value in metrics.items():
                if key != "support":
                    report_dict[label][key] = round(value * 100, 2)
                else:
                    report_dict[label][key] = int(value)


    df_report = pd.DataFrame(report_dict).transpose()

    # เตรียม context สำหรับ template
    context = {
        "accuracy": f"{acc * 100:.2f}%",
        "num_features": num_features,
        "duration": f"{time_duration:.2f}",
        "features": features_used,
        "params": params_used,
        "report_html": df_report.to_html(
            classes="table table-striped table-bordered",
            border=0,
            float_format="%.2f"
        )
    }

    # render ผ่าน jinja2
    env = Environment(loader=FileSystemLoader("templates"))
    template = env.get_template("report_template.html")
    html_out = template.render(context)

    # save HTML
    report_path = os.path.join(output_folder, "classification_report.html")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(html_out)

    print(f"📑 HTML report saved to {report_path}")

    # save model
    model_path = os.path.join(output_folder, 'xgboost-model.pkl')
    joblib.dump(model, model_path)
    print(f"💾 Model saved to {model_path}")

if __name__ == '__main__':
    if len(sys.argv) != 2:
        print("วิธีใช้: python training-ml-xgboost.py <output_folder>")
        sys.exit(1)
    output_folder = sys.argv[1]
    main(output_folder)
