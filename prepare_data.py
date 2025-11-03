import pandas as pd
import os, sys, glob, ipaddress
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

# -------------------------------------
# 📥 โหลด CSV จาก input folder หรือไฟล์เดี่ยว
# -------------------------------------
def load_csv(input_path, keep_fields):
    if os.path.isfile(input_path):
        files = [input_path]
    else:
        files = glob.glob(os.path.join(input_path, "*.csv"))

    if not files:
        sys.exit("❌ No CSV Files found.")

    dfs = []
    for f in files:
        print(f"📄 Loading file:", os.path.basename(f))
        df = pd.read_csv(f, on_bad_lines="skip")
        for col in keep_fields:
            if col not in df.columns:
                df[col] = None
        df = df[keep_fields]
        dfs.append(df)

    print(f"✅ Loaded {len(dfs)} file(s) successfully.")
    return pd.concat(dfs, ignore_index=True)


# -------------------------------------
# 🧩 แปลง IP → Octets
# -------------------------------------
def ip_to_octets(ip):
    try:
        parts = str(ip).split(".")
        if len(parts) == 4:
            return [int(p) if p.isdigit() else 0 for p in parts]
    except:
        pass
    return [0, 0, 0, 0]


# -------------------------------------
# 🧠 ฟังก์ชันหลัก: ทำความสะอาด + แปลงฟีเจอร์
# -------------------------------------
def transform_data(df, mode="auto"):
    df = df.copy().fillna("-")

    # ========= 1️⃣ แปลง IP =========
    src_octets = df["source.ip"].apply(ip_to_octets)
    dest_octets = df["destination.ip"].apply(ip_to_octets)
    df[[f"source_ip_oct{i}" for i in range(1, 5)]] = pd.DataFrame(src_octets.tolist(), index=df.index)
    df[[f"destination_ip_oct{i}" for i in range(1, 5)]] = pd.DataFrame(dest_octets.tolist(), index=df.index)

    # ========= 2️⃣ TF-IDF จาก URL =========
    whitelist_tokens = ["login", "admin", "update", "download", "upload",
                        "passwd", "config", "reset", "token", "php"]
    vectorizer = TfidfVectorizer(vocabulary=whitelist_tokens, token_pattern=r"[a-zA-Z]{3,}")
    url_features = vectorizer.fit_transform(df["url.original"].astype(str))
    url_df = pd.DataFrame(url_features.toarray(), columns=vectorizer.get_feature_names_out())

    # ========= 3️⃣ Time & HTTP Features =========
    df["@timestamp"] = pd.to_datetime(df["@timestamp"], errors="coerce")
    df["hour"] = df["@timestamp"].dt.hour.fillna(0).astype(int)
    df["weekday"] = df["@timestamp"].dt.weekday.fillna(0).astype(int)
    df["status_code"] = pd.to_numeric(df["http.response.status_code"], errors="coerce").fillna(0).astype(int)
    df["is_error"] = (df["status_code"] >= 400).astype(int)
    df["url_length"] = df["url.original"].astype(str).str.len()
    df["num_special_chars"] = df["url.original"].astype(str).str.count(r"[?=&%]")
    df["contains_suspicious_keyword"] = df["url.original"].astype(str).str.contains(
        "login|admin|cmd|token|download|shell", case=False, na=False
    ).astype(int)
    df["is_night"] = df["hour"].apply(lambda x: 1 if x <= 5 or x >= 22 else 0)

    # ========= 4️⃣ IP Behavior =========
    def is_internal(ip):
        try:
            return ipaddress.ip_address(ip).is_private
        except:
            return False

    df["src_is_private_ip"] = df["source.ip"].apply(is_internal).astype(int)
    df["dst_is_internal_ip"] = df["destination.ip"].apply(is_internal).astype(int)
    df["dst_is_public_ip"] = (df["dst_is_internal_ip"] == 0).astype(int)
    df["ip_match_local"] = (
        df["source.ip"].astype(str).str.split(".").str[0] ==
        df["destination.ip"].astype(str).str.split(".").str[0]
    ).astype(int)

    # ========= 5️⃣ Protocol & UA Behavior =========
    df["is_common_port"] = df["destination.port"].astype(str).isin(["80", "443", "8080"]).astype(int)
    df["protocol_is_http"] = df["network.protocol"].astype(str).str.contains("http", case=False, na=False).astype(int)
    df["req_method_is_post"] = df["http.request.method"].astype(str).str.upper().eq("POST").astype(int)
    df["is_referrer_missing"] = df["http.request.referrer"].astype(str).str.strip().isin(["-", "", "none"]).astype(int)
    df["same_country"] = (
        (df["source.geoip.country_code2"].astype(str).str.upper() ==
         df["destination.geoip.country_code2"].astype(str).str.upper()) &
        (df["source.geoip.country_code2"] != "-")
    ).astype(int)

    # ========= 6️⃣ User-Agent Intelligence =========
    ua_col = df["user_agent.original"].astype(str).str.lower()
    df["ua_is_empty"] = (ua_col == "-").astype(int)
    df["ua_is_browser"] = ua_col.str.contains("mozilla|chrome|safari|edge|firefox", na=False).astype(int)
    df["ua_is_microsoft"] = ua_col.str.contains("microsoft|windows|cryptoapi|msftconnect|delivery-optimization", na=False).astype(int)
    df["ua_is_python_script"] = ua_col.str.contains("python|requests|urllib|aiohttp", na=False).astype(int)
    df["ua_is_openstack"] = ua_col.str.contains("magnum|keystoneauth|openstack", na=False).astype(int)
    df["ua_is_cloud_service"] = ua_col.str.contains("aws|google|gcp|azure|cloudflare", na=False).astype(int)
    df["ua_is_bot"] = ua_col.str.contains("bot|crawler|curl", na=False).astype(int)
    df["ua_is_windows_update"] = ua_col.str.contains("microsoft|windows", na=False).astype(int)

    # ========= 7️⃣ Suspicious Pattern =========
    df["is_http_external"] = ((df["protocol_is_http"] == 1) & (df["dst_is_internal_ip"] == 0)).astype(int)
    df["is_suspicious_http"] = ((df["is_http_external"] == 1) & (df["ua_is_empty"] == 1)).astype(int)
    df["is_python_to_external"] = ((df["ua_is_python_script"] == 1) & (df["dst_is_internal_ip"] == 0)).astype(int)
    df["is_openstack_internal"] = ((df["ua_is_openstack"] == 1) & (df["dst_is_internal_ip"] == 1)).astype(int)

    # ========= 8️⃣ Microsoft Whitelist =========
    def is_microsoft_system(ua, dest, url):
        ua = str(ua).lower()
        dest = str(dest).lower()
        url = str(url).lower()
        ms_keywords = [
            "microsoft", "windows", "msftconnect", "cryptoapi",
            "delivery-optimization", "tlu.dl.delivery.mp.microsoft.com",
            "delivery.mp.microsoft.com", "officecdn", "windowsupdate",
            "update", "microsoft.com", "msedge.net"
        ]
        return any(k in ua for k in ms_keywords) or any(k in dest for k in ms_keywords) or any(k in url for k in ms_keywords)

    df["ua_is_microsoft_system"] = df.apply(
        lambda x: is_microsoft_system(x["user_agent.original"], x["destination.ip"], x["url.original"]),
        axis=1
    ).astype(int)

    def dest_is_microsoft_domain(dest):
        if not isinstance(dest, str):
            return False
        dest = dest.lower()
        microsoft_domains = [
            ".microsoft.com", ".windowsupdate.com", ".msedge.net",
            ".delivery.mp.microsoft.com", ".officecdn.microsoft.com"
        ]
        return any(dest.endswith(k) or k in dest for k in microsoft_domains)

    df["dest_is_microsoft"] = df["destination.ip"].apply(dest_is_microsoft_domain).astype(int)

    # ========= 9️⃣ Non-Browser External =========
    df["is_non_browser_external"] = (
        (df["ua_is_browser"] == 0) &
        (df["dst_is_internal_ip"] == 0) &
        (df["protocol_is_http"] == 1)
    ).astype(int)

    # ========= 🔟 Risk Scoring =========
    df["risk_score"] = (
        (df["is_suspicious_http"] * 2)
        + (df["is_python_to_external"] * 3)
        + (df["is_non_browser_external"] * 2)
        + (df["is_http_external"] * 1)
        - (df["is_openstack_internal"] * 2)
    ).clip(lower=0, upper=10)

    # ========= 🔟 รวมทั้งหมด =========
    final_df = pd.concat([
        df[[
            "status_code", "hour", "weekday", "is_error", "url_length",
            "num_special_chars", "contains_suspicious_keyword", "is_night",
            "src_is_private_ip", "dst_is_public_ip", "ip_match_local",
            "is_common_port", "protocol_is_http", "req_method_is_post",
            "is_referrer_missing", "same_country",
            "ua_is_browser", "ua_is_microsoft", "ua_is_python_script",
            "ua_is_openstack", "ua_is_cloud_service", "ua_is_bot",
            "ua_is_windows_update", "ua_is_microsoft_system", "dest_is_microsoft",
            "is_non_browser_external", "is_http_external", "is_suspicious_http",
            "is_python_to_external", "is_openstack_internal", "risk_score"
        ]],
        df[[f"source_ip_oct{i}" for i in range(1, 5)] +
           [f"destination_ip_oct{i}" for i in range(1, 5)]],
        url_df
    ], axis=1)

    # ======== 🧩 Label Handling (patched) ========
    if mode == "auto":
        mode = "train" if "ioc.dest_ip_misp_is_alert" in df.columns else "predict"

    if mode == "train" and "ioc.dest_ip_misp_is_alert" in df.columns:
        df["ioc.dest_ip_misp_is_alert"] = df["ioc.dest_ip_misp_is_alert"].fillna(0)
        final_df["label"] = df["ioc.dest_ip_misp_is_alert"].astype(int)
    elif mode == "predict":
        for col in ["ioc.dest_ip_misp_is_alert", "label"]:
            if col in final_df.columns:
                final_df = final_df.drop(columns=[col])
            
    

    return final_df


# -------------------------------------
# 🚀 main() สำหรับ training mode
# -------------------------------------
def main():
    input_folder = sys.argv[1]
    output_folder = sys.argv[2]
    test_pct = int(os.getenv("TEST_SET_PCT", 20))

    keep_fields = [
        "@timestamp", "source.ip", "destination.ip", "url.original",
        "http.response.status_code", "destination.port", "network.protocol",
        "user_agent.original", "http.request.method", "http.request.referrer",
        "source.geoip.country_code2", "destination.geoip.country_code2",
        "ioc.dest_ip_misp_is_alert"
    ]

    os.makedirs(output_folder, exist_ok=True)
    df = load_csv(input_folder, keep_fields)
    script_path = os.path.join(input_folder, "script_attacks.csv")
    if os.path.exists(script_path):
        print("⚡ Merging script_attacks.csv into dataset...")
        df_script = pd.read_csv(script_path, on_bad_lines="skip")
        for col in keep_fields:
            if col not in df_script.columns:
                df_script[col] = None
        df = pd.concat([df, df_script[keep_fields]], ignore_index=True)
    else:
        print("⚠️ script_attacks.csv not found — skipping merge")
        
    df_transformed = transform_data(df, mode="train")

    print("✅ Dataset size:", df_transformed.shape)
    if "label" in df_transformed.columns:
        print(df_transformed["label"].value_counts())

    train_df, test_df = train_test_split(
        df_transformed,
        test_size=test_pct/100,
        random_state=42,
        stratify=df_transformed["label"] if "label" in df_transformed else None
    )

    train_df.to_csv(os.path.join(output_folder, "training-set.csv"), index=False)
    test_df.to_csv(os.path.join(output_folder, "testing-set.csv"), index=False)
    print("✅ Data saved successfully.")


if __name__ == "__main__":
    main()
