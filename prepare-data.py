import pandas as pd
import os, sys, glob
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
import ipaddress

def load_csv(input_folder, keep_fields):
    files = glob.glob(os.path.join(input_folder, '*.csv'))
    if not files:
        sys.exit('There is no CSV Files')
    my_df = []
    for f in files:
        df = pd.read_csv(f, on_bad_lines='skip', usecols=keep_fields)
        my_df.append(df)
    return pd.concat(my_df, ignore_index=True)

def ip_to_octets(ip):
    try:
        parts = str(ip).split(".")
        if len(parts) == 4:
            octets = []
            for p in parts:
                try:
                    octets.append(int(p))
                except:
                    octets.append(0)
            return octets
        else:
            return [0, 0, 0, 0]  # กรณีไม่ใช่ IPv4
    except:
        return [0, 0, 0, 0]      # กรณี error อื่น ๆ


def transform_data(df):
    # 1) แปลง source.ip → src_oct1..src_oct4 แปลง destination.ip → dst_oct1..dst_oct4
    src_octets = df['source.ip'].apply(ip_to_octets)
    df[["source_ip_oct1", "source_ip_oct2", "source_ip_oct3", "source_ip_oct4"]] = pd.DataFrame(src_octets.tolist(), index=df.index)
    dest_octets = df['destination.ip'].apply(ip_to_octets)
    df[["destination_ip_oct1", "destination_ip_oct2", "destination_ip_oct3", "destination_ip_oct4"]] = pd.DataFrame(dest_octets.tolist(), index=df.index)

    # 2) แปลง url.original ด้วย TF-IDF โดยใช้ whitelist vocabulary
    whitelist_tokens = ["login", "admin", "update", "download", "upload", 
                        "passwd", "config", "reset", "token", "php"]
    vectorizer = TfidfVectorizer(vocabulary=whitelist_tokens, token_pattern=r'[a-zA-Z]{3,}')
    url_features = vectorizer.fit_transform(df["url.original"].astype(str))
    url_df = pd.DataFrame(url_features.toarray(), columns=vectorizer.get_feature_names_out())

    # 3) แปลง http.response.status_code → int
    # 3) แปลง http.response.status_code → int (handle non-numeric เช่น '-')
    df["status_code"] = pd.to_numeric(df["http.response.status_code"], errors="coerce").fillna(0).astype(int)


    # 4) Extract time features จาก @timestamp
    df["@timestamp"] = pd.to_datetime(df["@timestamp"], errors="coerce", format="%b %d, %Y @ %H:%M:%S.%f")
    df["hour"] = df["@timestamp"].dt.hour.fillna(0).astype(int)
    df["weekday"] = df["@timestamp"].dt.weekday.fillna(0).astype(int)

    # 5) รวม features ทั้งหมด
    final_df = pd.concat(
        [df[[
            "source_ip_oct1", "source_ip_oct2", "source_ip_oct3", "source_ip_oct4",
            "destination_ip_oct1", "destination_ip_oct2", "destination_ip_oct3", "destination_ip_oct4",
            "status_code", "hour", "weekday"
        ]],
        url_df],
        axis=1
    )

    # สร้าง label
    final_df['label'] = df['ioc.dest_ip_misp_is_alert'].astype(int)
    return final_df

def main():
    input_folder = sys.argv[1]
    output_folder = sys.argv[2]
    test_pct = int(os.getenv("TEST_SET_PCT", 20))

    # keep_fields เอาเฉพาะฟีลด์ที่ต้องใช้
    keep_fields = ['@timestamp', 
                   'source.ip', 
                   'destination.ip', 
                   'url.original', 
                   'http.response.status_code',
                   'ioc.dest_ip_misp_is_alert']  
    os.makedirs(output_folder, exist_ok=True)
    df = load_csv(input_folder, keep_fields)

    df_transform = transform_data(df)
    train_df, test_df = train_test_split(df_transform, test_size=test_pct/100, random_state=42,stratify=df_transform["label"])

    train_path = os.path.join(output_folder, 'training-set.csv')
    test_path = os.path.join(output_folder, 'testing-set.csv')

    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)
    print("✅ Data saved")

if __name__ == '__main__':
    main()
