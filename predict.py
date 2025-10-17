import os, sys
import glob
import pandas as pd
import joblib
import time
import datetime
import shutil #สำหรับทำ Archive
from sklearn.metrics import classification_report, accuracy_score
from jinja2 import Environment, FileSystemLoader
from prepare_data import transform_data

def get_latest_csv(input_folder):
    csv_files = glob.glob(os.path.join(input_folder, "*.csv"))
    if not csv_files:
        sys.exit("No CSV files found in input folder.")
    latest_file = max(csv_files, key=os.path.getmtime)
    return latest_file

def main():
    if len(sys.argv) < 4:
        sys.exit("Usage: python predict.py <model_path> <input_folder> <output_html>")

    model_path = sys.argv[1]
    input_folder = sys.argv[2]
    output_path = sys.argv[3]
    
    latest_csv = get_latest_csv(input_folder)


    print(f"📂 Using model: {model_path}")
    print(f"📥 Using input file: {latest_csv}")

    # โหลดข้อมูล log ใหม่ (raw log)
    df = pd.read_csv(latest_csv, on_bad_lines='skip')
    print(f"🔢 จำนวนข้อมูลทั้งหมด: {len(df)}")

    # Clean & transform raw log 
    print("🧹 Cleaning & transforming features ...")
    df_clean = transform_data(df)
    # โหลดโมเดล
    print("🤖 Loading trained model ...")
    model = joblib.load(model_path)

    # ตรวจว่ามีคอลัมน์ label หรือไม่
    if "label" in df_clean.columns:
        x_data = df_clean.drop(columns=["label"])
        y_true = df_clean["label"]
        labeled = True
    else:
        x_data = df_clean
        y_true = None
        labeled = False
        print("⚠️ No 'label' column found — running in unlabeled prediction mode.")

    # เริ่มทำนาย
    print("🔮 Predicting ...")
    start_time = time.time()
    y_pred = model.predict(x_data)
    end_time = time.time()
    duration = end_time - start_time

    # บันทึกผลการทำนายลง CSV
    df_result = df.copy()
    df_result["prediction"] = y_pred
    os.makedirs("data/output", exist_ok=True)
    df_result.to_csv("data/output/predict_result.csv", index=False)
    print("💾 Saved predictions to data/output/predict_result.csv")

    # สร้าง report
    if labeled:
        acc = accuracy_score(y_true, y_pred)
        print(f"✅ Accuracy: {acc * 100:.2f}%")

        report_dict = classification_report(y_true, y_pred, output_dict=True)
        for label, metrics in report_dict.items():
            if isinstance(metrics, dict):
                for key, value in metrics.items():
                    if key != "support":
                        report_dict[label][key] = round(value * 100, 2)
                    else:
                        report_dict[label][key] = int(value)
        df_report = pd.DataFrame(report_dict).transpose()
        report_html = df_report.to_html(
            classes="table table-striped table-bordered",
            border=0,
            float_format="%.2f"
        )
    else:
        acc = None
        report_html = "<p>No ground truth labels available for evaluation.</p>"

    # render HTML ด้วย template
    context = {
        "accuracy": f"{acc * 100:.2f}%" if acc is not None else "N/A",
        "duration": f"{duration:.2f}",
        "params": {},
        "report_html": report_html,
    }

    env = Environment(loader=FileSystemLoader("templates"))
    template = env.get_template("report_predict_template.html")
    html_out = template.render(context)

    # เขียนไฟล์ผลลัพธ์
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html_out)

    print(f"📑 HTML report saved to: {output_path}")
    print(f"⏱ Finished in {duration:.2f} seconds")
    
    archive_dir = os.path.join(input_folder,"archive")
    os.makedirs(archive_dir,exist_ok=True)
    shutil.move(latest_csv, os.path.join(archive_dir, os.path.basename(latest_csv)))
    
    # ✅ Logging
    output_folder = os.path.dirname(output_path)
    log_file = os.path.join(output_folder, "archive_log.txt")
    accuracy_value = f"{acc*100:.2f}%" if acc is not None else "N/A"

    with open(log_file, "a", encoding="utf-8") as log:
        log.write(
            f"[{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] "
            f"Predicted: {os.path.basename(latest_csv)}, "
            f"Accuracy: {accuracy_value}, "
            f"Rows: {len(df)}, "
            f"Duration: {duration:.2f} sec\n"
        )

if __name__ == "__main__":
    main()
