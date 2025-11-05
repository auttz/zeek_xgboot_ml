import os, sys, glob, time, datetime, shutil, ipaddress
import pandas as pd
import joblib
from minio import Minio
from sklearn.metrics import classification_report, accuracy_score
from jinja2 import Environment, FileSystemLoader
from prepare_data import transform_data

# Global Path Settings
BASE_OUTPUT_DIR = os.getenv("OUTPUT_DIR", "data/output")
os.makedirs(BASE_OUTPUT_DIR, exist_ok=True)

# ค้นหา CSV ล่าสุด
def get_latest_csv(input_folder):
    csv_files = glob.glob(os.path.join(input_folder, "*.csv")) + glob.glob(os.path.join(input_folder, "*.CSV"))
    if not csv_files:
        sys.exit(f"❌ No CSV files found in input folder: {input_folder}")
    latest = max(csv_files, key=os.path.getmtime)
    print(f"🕒 Latest CSV selected: {os.path.basename(latest)}")
    return latest

# โหลดข้อมูล + เตรียมฟีเจอร์
def load_and_prepare_data(latest_csv):
    print(f"📥 Loading: {latest_csv}")
    df = pd.read_csv(latest_csv, on_bad_lines='skip')
    print(f"🔢 Total rows: {len(df)}")
    print("🧹 Transforming features ...")
    df_clean = transform_data(df)
    return df, df_clean

# พยากรณ์และสร้างรายงาน
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

    # Predict with probability threshold
    print("🔮 Predicting with probability threshold ...")
    start = time.time()
    probs = model.predict_proba(x_data)
    THRESHOLD = 0.65
    y_pred = (probs[:, 1] >= THRESHOLD).astype(int)
    duration = time.time() - start

    df_result = df.copy()
    df_result["prob_1"] = probs[:, 1]
    df_result["prediction"] = y_pred

    # Whitelist Filtering
    def is_internal_ip(ip):
        try: return ipaddress.ip_address(ip).is_private
        except: return False

    def is_whitelisted(row):
        ua = str(row.get("user_agent.original", "")).lower()
        url = str(row.get("url.original", "")).lower()
        src_ip = str(row.get("source.ip", ""))
        dest_ip = str(row.get("destination.ip", ""))
        status = str(row.get("http.response.status_code", ""))

        ms_keywords = ["msftconnecttest", "microsoft", "windows update", "cryptoapi", "windowsupdate", "officecdn", "outlook", "onenote", "onedrive", "bingbot", "defender", "edge"]
        safe_domains = ["microsoft.com", "windows.com", "office.com", "msedge.net", "live.com", "bing.com", "skype.com", "update.microsoft.com", "google.com", "youtube.com", "apple.com", "icloud.com", "cloudflare.com", "akamai.net"]

        if any(k in ua for k in ms_keywords) or any(d in url for d in safe_domains):
            if is_internal_ip(src_ip) or status.startswith("20") or "http" in url:
                return True
        if any(k in ua for k in ["mozilla", "chrome", "safari", "applewebkit"]):
            if is_internal_ip(src_ip): return True
        return False

    df_result["is_whitelist"] = df_result.apply(is_whitelisted, axis=1)
    whitelist_count = df_result["is_whitelist"].sum()

    if whitelist_count > 0:
        print(f"🧩 Found {whitelist_count} whitelisted benign logs (Microsoft/System).")
        whitelist_path = os.path.join(BASE_OUTPUT_DIR, "whitelist_filtered.csv")
        df_result[df_result["is_whitelist"] == True].to_csv(whitelist_path, index=False)
        print(f"💾 Whitelist entries saved → {whitelist_path}")
        df_result.loc[df_result["is_whitelist"] == True, "prediction"] = 0

    # Post-filter ลด False Positive เพิ่มเติม
    def post_filter(row, pred):
        if pred == 1:
            proto = str(row.get("network.protocol", "")).lower()
            code = str(row.get("http.response.status_code", ""))
            ua = str(row.get("user_agent.original", "")).lower()
            if proto == "https" and code.startswith("20"):
                if "mozilla" in ua or "chrome" in ua or "safari" in ua: return 0
        src_cc = str(row.get("source.geoip.country_code2", ""))
        dst_cc = str(row.get("destination.geoip.country_code2", ""))
        if pred == 1 and src_cc and dst_cc and src_cc == dst_cc: return 0
        if pred == 1 and row.get("prob_1", 0) < 0.55: return 0
        return pred

    df_result["prediction"] = df_result.apply(lambda r: post_filter(r, r["prediction"]), axis=1)

    # บันทึกผลลัพธ์
    output_csv_path = os.path.join(BASE_OUTPUT_DIR, "predict_result.csv")
    df_result.to_csv(output_csv_path, index=False)
    print(f"💾 Saved predictions → {output_csv_path}")

    # สรุปจำนวนผลลัพธ์
    total_logs = len(df_result)
    alerts = int((df_result["prediction"] == 1).sum())
    normals = total_logs - alerts
    print(f"📊 Summary: Total={total_logs} | Whitelist={whitelist_count} | Alerts(after filter)={alerts} | Normal={normals}")

    # Accuracy (ถ้ามี label)
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

# สร้าง HTML Report
def generate_html_report(acc, duration, report_html):
    env = Environment(loader=FileSystemLoader("templates"))
    template = env.get_template("report_predict_template.html")
    context = {"accuracy": f"{acc*100:.2f}%" if acc else "N/A", "duration": f"{duration:.2f}", "params": {}, "report_html": report_html}
    output_html_path = os.path.join(BASE_OUTPUT_DIR, "classification_report_predict.html")
    html_out = template.render(context)
    with open(output_html_path, "w", encoding="utf-8") as f:
        f.write(html_out)
    print(f"📑 HTML report saved → {output_html_path}")
    return output_html_path

# Upload ขึ้น MinIO
def upload_to_minio():
    try:
        client = Minio(os.getenv("MINIO_ENDPOINT", "localhost:9000"), access_key=os.getenv("MINIO_ACCESS_KEY", "admin"), secret_key=os.getenv("MINIO_SECRET_KEY", "12345678"), secure=False)
        bucket_name = "zeek-data"
        if not client.bucket_exists(bucket_name):
            client.make_bucket(bucket_name)
            print(f"✅ Created bucket: {bucket_name}")
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        print("\n📤 Uploading to MinIO:")
        report_path = os.path.join(BASE_OUTPUT_DIR, "classification_report_predict.html")
        if os.path.exists(report_path):
            obj_path = f"reports/{timestamp}/classification_report_predict.html"
            print(f"→ {obj_path}")
            client.fput_object(bucket_name, obj_path, report_path)
        csv_path = os.path.join(BASE_OUTPUT_DIR, "predict_result.csv")
        if os.path.exists(csv_path):
            obj_path = f"datasets/{timestamp}/predict_result.csv"
            print(f"→ {obj_path}")
            client.fput_object(bucket_name, obj_path, csv_path)
        print("✅ Upload complete!\n")
    except Exception as e:
        print(f"❌ Upload failed: {e}")

# Archive และบันทึก log
def archive_and_log(latest_csv, input_folder, acc, duration, df_len):
    archive_dir = os.path.join(input_folder, "archive")
    os.makedirs(archive_dir, exist_ok=True)
    shutil.move(latest_csv, os.path.join(archive_dir, os.path.basename(latest_csv)))
    log_file = os.path.join(BASE_OUTPUT_DIR, "archive_log.txt")
    with open(log_file, "a", encoding="utf-8") as log:
        log.write(f"[{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Predicted: {os.path.basename(latest_csv)}, Accuracy: {f'{acc*100:.2f}%' if acc else 'N/A'}, Rows: {df_len}, Duration: {duration:.2f} sec\n")
    print("🗃 Archived input file & updated log.")

# MAIN
def main():
    if len(sys.argv) < 4:
        sys.exit("Usage: python predict.py <model_path> <input_folder> <output_html>")
    model_path, input_folder, _ = sys.argv[1:4]
    model_path = os.path.abspath(model_path)
    input_folder = os.path.abspath(input_folder)
    print(f"🧭 Model path: {model_path}")
    print(f"📂 Input folder: {input_folder}")
    latest_csv = get_latest_csv(input_folder)
    df, df_clean = load_and_prepare_data(latest_csv)
    y_pred, acc, report_html, duration = run_prediction(model_path, df, df_clean)
    html_output_path = generate_html_report(acc, duration, report_html)
    upload_to_minio()
    archive_and_log(latest_csv, input_folder, acc, duration, len(df))
    print(f"✅ Finished successfully in {duration:.2f} seconds.")

if __name__ == "__main__":
    main()
