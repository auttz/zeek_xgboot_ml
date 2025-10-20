import os, sys, glob, time, datetime, shutil
import pandas as pd
import joblib
from minio import Minio
from sklearn.metrics import classification_report, accuracy_score
from jinja2 import Environment, FileSystemLoader
from prepare_data import transform_data


# -------------------------------------------
# 1️⃣ Helper: ค้นหา CSV ล่าสุด
# -------------------------------------------
def get_latest_csv(input_folder):
    csv_files = glob.glob(os.path.join(input_folder, "*.csv"))
    if not csv_files:
        sys.exit("❌ No CSV files found in input folder.")
    return max(csv_files, key=os.path.getmtime)


# -------------------------------------------
# 2️⃣ โหลดข้อมูล + เตรียมฟีเจอร์
# -------------------------------------------
def load_and_prepare_data(latest_csv):
    print(f"📥 Loading: {latest_csv}")
    df = pd.read_csv(latest_csv, on_bad_lines='skip')
    print(f"🔢 Total rows: {len(df)}")

    print("🧹 Transforming features ...")
    df_clean = transform_data(df)
    return df, df_clean

# -------------------------------------------
# 3️⃣ พยากรณ์และสร้างรายงาน
# -------------------------------------------
def run_prediction(model_path, df, df_clean):
    print("🤖 Loading trained model ...")
    model = joblib.load(model_path)

    if "label" in df_clean.columns:
        x_data = df_clean.drop(columns=["label"])
        y_true = df_clean["label"]
        labeled = True
    else:
        x_data = df_clean
        y_true = None
        labeled = False
        print("⚠️ No 'label' column found — running in unlabeled mode.")

    print("🔮 Predicting ...")
    start = time.time()
    y_pred = model.predict(x_data)
    duration = time.time() - start

    df_result = df.copy()
    df_result["prediction"] = y_pred
    os.makedirs("data/output", exist_ok=True)
    df_result.to_csv("data/output/predict_result.csv", index=False)
    print("💾 Saved predictions → data/output/predict_result.csv")

    acc, report_html = None, "<p>No ground truth labels available.</p>"
    if labeled:
        acc = accuracy_score(y_true, y_pred)
        print(f"✅ Accuracy: {acc*100:.2f}%")

        report_dict = classification_report(y_true, y_pred, output_dict=True)
        for lbl, metrics in report_dict.items():
            if isinstance(metrics, dict):
                for k, v in metrics.items():
                    metrics[k] = round(v * 100, 2) if k != "support" else int(v)
        df_report = pd.DataFrame(report_dict).transpose()
        report_html = df_report.to_html(classes="table table-striped table-bordered", border=0)

    return y_pred, acc, report_html, duration


# -------------------------------------------
# 4️⃣ สร้าง HTML Report
# -------------------------------------------
def generate_html_report(acc, duration, report_html, output_path):
    env = Environment(loader=FileSystemLoader("templates"))
    template = env.get_template("report_predict_template.html")

    context = {
        "accuracy": f"{acc*100:.2f}%" if acc else "N/A",
        "duration": f"{duration:.2f}",
        "params": {},
        "report_html": report_html,
    }

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    html_out = template.render(context)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html_out)
    print(f"📑 HTML report saved → {output_path}")
    


# -------------------------------------------
# 5️⃣ Upload ขึ้น MinIO
# -------------------------------------------
def upload_to_minio(output_path):
    try:
        client = Minio(
            os.getenv("MINIO_ENDPOINT", "localhost:9000"),
            access_key=os.getenv("MINIO_ACCESS_KEY", "admin"),
            secret_key=os.getenv("MINIO_SECRET_KEY", "12345678"),
            secure=False
        )

        bucket_name = "zeek-data"
        if not client.bucket_exists(bucket_name):
            client.make_bucket(bucket_name)
            print(f"✅ Created bucket: {bucket_name}")

        timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        print("\n📤 Uploading to MinIO:")

        # 🧾 HTML Report
        report_path = f"reports/{timestamp}/classification_report_predict.html"
        print(f"→ {report_path}")
        client.fput_object(bucket_name, report_path, output_path)

        # 📊 CSV
        csv_path = "data/output/predict_result.csv"
        if os.path.exists(csv_path):
            csv_obj = f"datasets/{timestamp}/predict_result.csv"
            print(f"→ {csv_obj}")
            client.fput_object(bucket_name, csv_obj, csv_path)

        # 🧠 Model
        model_path = "data/output/xgboost-model.pkl"
        if os.path.exists(model_path):
            model_obj = f"models/{timestamp}/xgboost-model.pkl"
            print(f"→ {model_obj}")
            client.fput_object(bucket_name, model_obj, model_path)

        # 🗒️ Log
        log_path = "data/output/archive_log.txt"
        if os.path.exists(log_path):
            log_obj = f"logs/{timestamp}/archive_log.txt"
            print(f"→ {log_obj}")
            client.fput_object(bucket_name, log_obj, log_path)

        print("✅ Upload complete!\n")

    except Exception as e:
        print(f"❌ Upload failed: {e}")
    
# -------------------------------------------
# 6️⃣ จัดการ archive และ log
# -------------------------------------------
def archive_and_log(latest_csv, input_folder, acc, duration, df_len):
    archive_dir = os.path.join(input_folder, "archive")
    os.makedirs(archive_dir, exist_ok=True)
    shutil.move(latest_csv, os.path.join(archive_dir, os.path.basename(latest_csv)))

    log_file = "data/output/archive_log.txt"
    with open(log_file, "a", encoding="utf-8") as log:
        log.write(
            f"[{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] "
            f"Predicted: {os.path.basename(latest_csv)}, "
            f"Accuracy: {f'{acc*100:.2f}%' if acc else 'N/A'}, "
            f"Rows: {df_len}, "
            f"Duration: {duration:.2f} sec\n"
        )
    print("🗃 Archived input file & updated log.")


# -------------------------------------------
# 🧩 MAIN PIPELINE
# -------------------------------------------
def main():
    if len(sys.argv) < 4:
        sys.exit("Usage: python predict.py <model_path> <input_folder> <output_html>")

    model_path, input_folder, output_path = sys.argv[1:4]

    latest_csv = get_latest_csv(input_folder)
    df, df_clean = load_and_prepare_data(latest_csv)
    y_pred, acc, report_html, duration = run_prediction(model_path, df, df_clean)
    generate_html_report(acc, duration, report_html, output_path)
    upload_to_minio(output_path)
    archive_and_log(latest_csv, input_folder, acc, duration, len(df))

    print(f"✅ Finished successfully in {duration:.2f} seconds.")


if __name__ == "__main__":
    main()
