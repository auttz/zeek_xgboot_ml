import pandas as pd
import os

alert_file = "data/input/alerts_full.csv"
normale_file = "data/input/normal_logs.csv"
output_file = "data/output/dataset_v2.csv"

os.makedirs("date/output",exist_ok=True)

df_alert = pd.read_csv(alert_file,on_bad_lines="skip")
df_normal = pd.read_csv(normale_file,on_bad_lines="skip")

df_alert["label"] = 1
df_normal["label"] = 0

# รวม dataset
df_all = pd.concat([df_alert, df_normal], ignore_index=True)
df_all.replace(["-", "None", "N/A"], pd.NA, inplace=True)
df_all.dropna(subset=["destination.ip"], inplace=True)

# สุ่มลำดับ (shuffle)
df_all = df_all.sample(frac=1, random_state=42).reset_index(drop=True)

df_all.to_csv(output_file, index=False)
print(f"✅ บันทึก dataset รวมแล้ว: {output_file}")
print(df_all["label"].value_counts())