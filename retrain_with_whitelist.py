import os
import pandas as pd
import subprocess

# -----------------------------
# 🌍 Path Settings
# -----------------------------
DATASET_BASE = "data"
OUTPUT_DIR = os.path.join(DATASET_BASE, "output")
WHITELIST_DIR = os.path.join(DATASET_BASE, "whitelist")

os.makedirs(WHITELIST_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# -----------------------------
# 1️⃣ Extract whitelist traffic
# -----------------------------
predict_file = os.path.join(OUTPUT_DIR, "predict_result.csv")
if not os.path.exists(predict_file):
    raise FileNotFoundError("❌ ไม่พบไฟล์ predict_result.csv กรุณารัน predict ก่อน retrain")

df = pd.read_csv(predict_file)

# เงื่อนไขดึง Microsoft / Windows traffic
whitelist_df = df[
    df["user_agent.original"].astype(str).str.contains("Microsoft|CryptoAPI|NCSI|Windows", case=False, na=False)
]

if whitelist_df.empty:
    print("⚠️ ไม่พบ Microsoft traffic ใน predict_result.csv")
else:
    whitelist_df["ioc.dest_ip_misp_is_alert"] = 0
    whitelist_path = os.path.join(WHITELIST_DIR, "whitelist_filtered.csv")
    whitelist_df.to_csv(whitelist_path, index=False)
    print(f"✅ Extracted whitelist: {len(whitelist_df)} rows saved → {whitelist_path}")

# -----------------------------
# 2️⃣ Merge whitelist กับ dataset เดิม
# -----------------------------
dataset_old = os.path.join(OUTPUT_DIR, "dataset_v3.csv")
dataset_new = os.path.join(OUTPUT_DIR, "dataset_v4.csv")

if not os.path.exists(dataset_old):
    raise FileNotFoundError("❌ ไม่พบ dataset_v3.csv กรุณาตรวจสอบไฟล์ dataset")

df_main = pd.read_csv(dataset_old)
df_merge = pd.concat([df_main, whitelist_df], ignore_index=True)
df_merge.to_csv(dataset_new, index=False)
print(f"✅ รวม dataset เสร็จสิ้น: {dataset_new} ({df_merge.shape[0]} rows)")


# -----------------------------
# 3️⃣ รัน prepare_data.py → output folder
# -----------------------------
print("\n🚀 Running prepare_data.py ...")
subprocess.run(["python", "prepare_data.py", dataset_new, OUTPUT_DIR], check=True)

# -----------------------------
# 4️⃣ รัน training script โดยใช้ไฟล์ใน output
# -----------------------------
print("\n🤖 Retraining model ...")
subprocess.run([
    "python", "training-ml-xgboost.py",
    OUTPUT_DIR
], check=True)


print("\n✅ Retrain completed successfully! 🎉")
print("🔁 ได้โมเดลใหม่ xgboost-model.pkl พร้อมใช้งานกับ predict.py แล้ว ✅")
