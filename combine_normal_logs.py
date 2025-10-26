import pandas as pd
import glob
import os

# รวมไฟล์ CSV ที่ชื่อขึ้นต้นว่า normal_
input_folder = "data/input"
pattern = os.path.join(input_folder, "normal_*.csv")

csv_files = glob.glob(pattern)
print(f"📂 พบไฟล์ทั้งหมด {len(csv_files)} ไฟล์")

dfs = [pd.read_csv(f, on_bad_lines="skip") for f in csv_files]
df = pd.concat(dfs, ignore_index=True)

# ล้างค่า
df.replace(["-", "None", "N/A"], pd.NA, inplace=True)
df.dropna(subset=["destination.ip"], inplace=True)

# บันทึกไฟล์รวม
output_file = os.path.join(input_folder, "normal_logs.csv")
df.to_csv(output_file, index=False)

print(f"✅ รวมไฟล์ทั้งหมดแล้ว: {output_file}")
print(f"📊 จำนวนข้อมูลทั้งหมด: {len(df)} rows")
