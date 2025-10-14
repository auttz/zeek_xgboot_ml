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
        df = pd.read_csv(f, on_bad_lines='skip')  # 👈 อ่านทั้งหมดก่อน
        # เติมคอลัมน์ที่หายไปให้ครบ
        for col in keep_fields:
            if col not in df.columns:
                df[col] = None
        # เลือกเฉพาะคอลัมน์ที่ต้องใช้
        df = df[keep_fields]
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
    # --------------------------------------------------------
    # 1) แปลง IP → octets
    # --------------------------------------------------------
    src_octets = df['source.ip'].apply(ip_to_octets)
    df[["source_ip_oct1", "source_ip_oct2", "source_ip_oct3", "source_ip_oct4"]] = pd.DataFrame(src_octets.tolist(), index=df.index)

    dest_octets = df['destination.ip'].apply(ip_to_octets)
    df[["destination_ip_oct1", "destination_ip_oct2", "destination_ip_oct3", "destination_ip_oct4"]] = pd.DataFrame(dest_octets.tolist(), index=df.index)

    # --------------------------------------------------------
    # 2) TF-IDF จาก url.original
    # --------------------------------------------------------
    whitelist_tokens = ["login", "admin", "update", "download", "upload", 
                        "passwd", "config", "reset", "token", "php"]
    vectorizer = TfidfVectorizer(vocabulary=whitelist_tokens, token_pattern=r'[a-zA-Z]{3,}')
    url_features = vectorizer.fit_transform(df["url.original"].astype(str))
    url_df = pd.DataFrame(url_features.toarray(), columns=vectorizer.get_feature_names_out())

    # --------------------------------------------------------
    # 3) แปลง http.response.status_code → int
    # --------------------------------------------------------
    df["status_code"] = pd.to_numeric(df["http.response.status_code"], errors="coerce").fillna(0).astype(int)

    # --------------------------------------------------------
    # 4) Extract time features จาก @timestamp
    # --------------------------------------------------------
    df["@timestamp"] = pd.to_datetime(df["@timestamp"], errors="coerce", format="%b %d, %Y @ %H:%M:%S.%f")
    df["hour"] = df["@timestamp"].dt.hour.fillna(0).astype(int)
    df["weekday"] = df["@timestamp"].dt.weekday.fillna(0).astype(int)

    # --------------------------------------------------------
    # 5) ฟีเจอร์เดิม 8 ตัว
    # --------------------------------------------------------

    # 5.1 is_error → สถานะ HTTP error code (>=400)
    df["is_error"] = 0
    for i in df.index:
        code = df.at[i, "status_code"]
        df.at[i, "is_error"] = 1 if code >= 400 else 0

    # 5.2 url_length → ความยาวของ URL
    df["url_length"] = 0
    for i in df.index:
        url = str(df.at[i, "url.original"])
        df.at[i, "url_length"] = len(url)

    # 5.3 num_special_chars → จำนวน ?, =, &, %
    df["num_special_chars"] = 0
    for i in df.index:
        url = str(df.at[i, "url.original"])
        count = url.count('?') + url.count('=') + url.count('&') + url.count('%')
        df.at[i, "num_special_chars"] = count

    # 5.4 contains_suspicious_keyword → มีคำต้องสงสัยไหม (login/admin/cmd/token/download/shell)
    suspicious_words = ['login', 'admin', 'cmd', 'token', 'download', 'shell']
    df["contains_suspicious_keyword"] = 0
    for i in df.index:
        url = str(df.at[i, "url.original"]).lower()
        found = 0
        for word in suspicious_words:
            if word in url:
                found = 1
                break
        df.at[i, "contains_suspicious_keyword"] = found

    # 5.5 is_night → request ตอนกลางคืน (0-5 หรือ >=22)
    df["is_night"] = 0
    for i in df.index:
        hour = df.at[i, "hour"]
        if hour <= 5 or hour >= 22:
            df.at[i, "is_night"] = 1

    # 5.6 src_is_private_ip → source.ip เป็น private IP หรือไม่
    df["src_is_private_ip"] = 0
    for i in df.index:
        ip_str = str(df.at[i, "source.ip"])
        if ip_str.startswith("10.") or ip_str.startswith("192.168.") or ip_str.startswith("172."):
            df.at[i, "src_is_private_ip"] = 1

    # 5.7 dst_is_public_ip → destination.ip เป็น public IP หรือไม่
    df["dst_is_public_ip"] = 0
    for i in df.index:
        ip_str = str(df.at[i, "destination.ip"])
        if not (ip_str.startswith("10.") or ip_str.startswith("192.168.") or ip_str.startswith("172.")):
            df.at[i, "dst_is_public_ip"] = 1

    # 5.8 ip_match_local → source และ destination อยู่ subnet เดียวกันไหม
    df["ip_match_local"] = 0
    for i in df.index:
        src_ip = str(df.at[i, "source.ip"]).split(".")
        dst_ip = str(df.at[i, "destination.ip"]).split(".")
        if len(src_ip) == 4 and len(dst_ip) == 4:
            if src_ip[0] == dst_ip[0]:
                df.at[i, "ip_match_local"] = 1

    # --------------------------------------------------------
    # 6) ฟีเจอร์พฤติกรรมใหม่อีก 8 ตัว
    # --------------------------------------------------------

    # 6.1 is_common_port → พอร์ตทั่วไป (80,443,8080)
    df["is_common_port"] = 0
    for i in df.index:
        port = str(df.at[i, "destination.port"])
        if port in ["80", "443", "8080"]:
            df.at[i, "is_common_port"] = 1

    # 6.2 protocol_is_http → protocol มีคำว่า http
    df["protocol_is_http"] = 0
    for i in df.index:
        proto = str(df.at[i, "network.protocol"]).lower()
        if "http" in proto:
            df.at[i, "protocol_is_http"] = 1

    # 6.3 ua_is_bot → user-agent มีคำว่า bot, crawler, curl, python
    df["ua_is_bot"] = 0
    for i in df.index:
        ua = str(df.at[i, "user_agent.original"]).lower()
        if any(bot in ua for bot in ["bot", "crawler", "curl", "python"]):
            df.at[i, "ua_is_bot"] = 1

    # 6.4 ua_is_windows_update → user-agent เป็นของ Windows/Microsoft
    df["ua_is_windows_update"] = 0
    for i in df.index:
        ua = str(df.at[i, "user_agent.original"]).lower()
        if "microsoft" in ua or "windows" in ua:
            df.at[i, "ua_is_windows_update"] = 1

    # 6.5 url_has_file_ext → URL มีนามสกุล .exe, .zip, .bat, .php, .js
    df["url_has_file_ext"] = 0
    for i in df.index:
        url = str(df.at[i, "url.original"]).lower()
        if any(ext in url for ext in [".exe", ".zip", ".bat", ".php", ".js"]):
            df.at[i, "url_has_file_ext"] = 1

    # 6.6 req_method_is_post → ใช้ HTTP POST หรือไม่
    df["req_method_is_post"] = 0
    for i in df.index:
        method = str(df.at[i, "http.request.method"]).upper()
        if method == "POST":
            df.at[i, "req_method_is_post"] = 1

    # 6.7 is_referrer_missing → ไม่มีค่า Referrer
    df["is_referrer_missing"] = 0
    for i in df.index:
        ref = str(df.at[i, "http.request.referrer"]).strip()
        if ref == "-" or ref == "" or ref.lower() == "none":
            df.at[i, "is_referrer_missing"] = 1

    # 6.8 same_country → source/destination มาจากประเทศเดียวกัน
    df["same_country"] = 0
    for i in df.index:
        src = str(df.at[i, "source.geoip.country_code2"]).upper()
        dst = str(df.at[i, "destination.geoip.country_code2"]).upper()
        if src == dst and src != "-" and src != "NONE":
            df.at[i, "same_country"] = 1

    # --------------------------------------------------------
    # 7) รวม features ทั้งหมด
    # --------------------------------------------------------
    final_df = pd.concat([
        df[[
            # IP octets
            "source_ip_oct1", "source_ip_oct2", "source_ip_oct3", "source_ip_oct4",
            "destination_ip_oct1", "destination_ip_oct2", "destination_ip_oct3", "destination_ip_oct4",
            # time/status
            "status_code", "hour", "weekday",
            # ฟีเจอร์เดิม
            "is_error", "url_length", "num_special_chars",
            "contains_suspicious_keyword", "is_night",
            "src_is_private_ip", "dst_is_public_ip", "ip_match_local",
            # ฟีเจอร์ใหม่
            "is_common_port", "protocol_is_http", "ua_is_bot", "ua_is_windows_update",
            "url_has_file_ext", "req_method_is_post", "is_referrer_missing", "same_country"
        ]],
        url_df
    ], axis=1)

    # --------------------------------------------------------
    # 8) สร้าง Label
    # --------------------------------------------------------
    final_df["label"] = df["ioc.dest_ip_misp_is_alert"].astype(int)

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
                    'destination.port',
                    'network.protocol',
                    'user_agent.original',
                    'http.request.method',
                    'http.request.referrer',
                    'source.geoip.country_code2',
                    'destination.geoip.country_code2',
                    'ioc.dest_ip_misp_is_alert']  
    
    os.makedirs(output_folder, exist_ok=True)
    df = load_csv(input_folder, keep_fields)
    #print(df.head())      # ดูข้อมูล 5 แถวแรก
    #print(df.tail())      # ดู 5 แถวสุดท้าย
    #print(df.shape)       # ดูจำนวน row, col

    df_transform = transform_data(df)
    print("✅ ขนาด dataset ทั้งหมด:", df_transform.shape)
    print(df_transform['label'].value_counts())

    train_df, test_df = train_test_split(df_transform, test_size=test_pct/100, random_state=42,stratify=df_transform["label"])
    print("✅ training-set:", train_df.shape)
    print("✅ testing-set:", test_df.shape)
    print(test_df['label'].value_counts())

    train_path = os.path.join(output_folder, 'training-set.csv')
    test_path = os.path.join(output_folder, 'testing-set.csv')

    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)
    print("✅ Data saved")

if __name__ == '__main__':
    main()
