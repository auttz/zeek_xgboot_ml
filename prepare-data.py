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

def ip_to_int(ip):
    try:
        return int(ipaddress.ip_address(ip))
    except:
        return 0

def transform_data(df):
    # 1) แปลง source/destination IP เป็นเลขจำนวนเต็ม
    df["source_ip"] = df["source.ip"].apply(ip_to_int)
    df["destination_ip"] = df["destination.ip"].apply(ip_to_int)

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
    df["@timestamp"] = pd.to_datetime(
        df["@timestamp"], 
        errors="coerce", 
        format="%b %d, %Y @ %H:%M:%S.%f"
    )
    df["hour"] = df["@timestamp"].dt.hour.fillna(0).astype(int)
    df["weekday"] = df["@timestamp"].dt.weekday.fillna(0).astype(int)

    # 5) รวม features ทั้งหมด
    final_df = pd.concat(
        [df[["source_ip", "destination_ip", "status_code", "hour", "weekday"]], url_df],
        axis=1
    )
    return final_df

def main():
    input_folder = sys.argv[1]
    output_folder = sys.argv[2]
    test_pct = int(os.getenv("TEST_SET_PCT", 20))

    # keep_fields เอาเฉพาะฟีลด์ที่ต้องใช้
    keep_fields = ['@timestamp', 'source.ip', 'destination.ip', 'url.original', 'http.response.status_code']  
    os.makedirs(output_folder, exist_ok=True)
    df = load_csv(input_folder, keep_fields)

    df_transform = transform_data(df)
    train_df, test_df = train_test_split(df_transform, test_size=test_pct/100, random_state=42)

    train_path = os.path.join(output_folder, 'training-set.csv')
    test_path = os.path.join(output_folder, 'testing-set.csv')

    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)

if __name__ == '__main__':
    main()
