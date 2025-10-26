import pandas as pd
import os

# path ต่างๆ
dataset_v2 = "data/output/dataset_v2.csv"
ncsi_file = "data/input/microsoft_ncsi_normal.csv"
output_file = "data/output/dataset_v3.csv"
os.makedirs("data/output", exist_ok=True)

print("📥 Loading dataset_v2.csv ...")
df_old = pd.read_csv(dataset_v2, on_bad_lines="skip", low_memory=False)

print("📥 Loading microsoft_ncsi_normal.csv ...")
# 🔧 เพิ่ม encoding และลองเช็ค delimiter อัตโนมัติ
try:
    df_ncsi = pd.read_csv(ncsi_file, encoding="utf-8-sig", on_bad_lines="skip", low_memory=False)
except Exception:
    df_ncsi = pd.read_csv(ncsi_file, encoding="utf-8-sig", on_bad_lines="skip", sep=";", low_memory=False)

# ใส่ label=0 เพราะเป็น traffic ปกติ
df_ncsi["label"] = 0

print(f"✅ dataset_v2 rows: {len(df_old)}")
print(f"✅ microsoft_ncsi_normal rows: {len(df_ncsi)}")

# รวมเข้าด้วยกัน
df_v3 = pd.concat([df_old, df_ncsi], ignore_index=True)
df_v3 = df_v3.sample(frac=1, random_state=42).reset_index(drop=True)

# บันทึก
df_v3.to_csv(output_file, index=False)
print(f"✅ Created dataset_v3.csv — total {len(df_v3)} rows")
