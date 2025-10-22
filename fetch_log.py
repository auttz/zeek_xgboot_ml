import os
import requests
import pandas as pd
import datetime
from dotenv import load_dotenv
from requests.auth import HTTPBasicAuth
import urllib3

# ปิด warning SSL (กรณีใช้ HTTPS แบบ self-signed)
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# โหลดค่าจากไฟล์ .env
load_dotenv("secret.env")

KIBANA_URL = os.getenv("KIBANA_URL")
KIBANA_USER = os.getenv("KIBANA_USER")
KIBANA_PASS = os.getenv("KIBANA_PASS")
INDEX_PATTERN = os.getenv("INDEX_PATTERN", "rtarf-events-beat-*")
FETCH_SIZE = int(os.getenv("FETCH_SIZE", "1000"))

OUTPUT_DIR = "data/newdata"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def test_connection():
    try:
        url = f"{KIBANA_URL}/_cluster/health"
        resp = requests.get(url, auth=HTTPBasicAuth(KIBANA_USER, KIBANA_PASS), verify=False, timeout=10)
        if resp.status_code == 200:
            print("✅ Connected to Elasticsearch cluster successfully.")
            return True
        else:
            print(f"❌ Connection failed: {resp.status_code} {resp.text}")
            return False
    except Exception as e:
        print(f"❌ Error connecting to Elasticsearch: {e}")
        return False


def fetch_logs():
    print("🚀 Fetching logs from Elasticsearch (last 15 minutes)...")

    query = {
    "size": FETCH_SIZE,
    "_source": [
        "@timestamp",
        "destination.port",
        "network.protocol",
        "user_agent.original",
        "http.request.method",
        "http.request.referrer",
        "source.geoip.country_code2",
        "destination.geoip.country_code2",
        "ioc.dest_ip_misp_is_alert"
    ],
    "query": {
        "bool": {
            "must": [
                {"term": {"event.dataset.keyword": "zeek.http"}},
                {
                    "range": {
                        "@timestamp": {
                            "gte": "now-15m",
                            "lt": "now"
                        }
                    }
                }
            ]
        }
    }
}


    url = f"{KIBANA_URL}/{INDEX_PATTERN}/_search"

    try:
        response = requests.post(
            url,
            auth=HTTPBasicAuth(KIBANA_USER, KIBANA_PASS),
            json=query,
            verify=False,
            timeout=30
        )

        if response.status_code != 200:
            print(f"❌ Failed to fetch logs: {response.status_code}")
            print(response.text)
            return None

        data = response.json()
        hits = data.get("hits", {}).get("hits", [])
        print(f"📦 Retrieved {len(hits)} logs")

        if not hits:
            print("⚠️ No logs found in the last 15 minutes.")
            return None

        records = [h["_source"] for h in hits]
        df = pd.DataFrame(records)

        # ตั้งชื่อไฟล์ตามเวลา
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = os.path.join(OUTPUT_DIR, f"fetched_logs_{timestamp}.csv")

        df.to_csv(output_file, index=False)
        print(f"✅ Saved logs to {output_file}")
        return output_file

    except requests.exceptions.ReadTimeout:
        print("⏳ Request timed out. Try increasing the time range or reducing FETCH_SIZE.")
    except requests.exceptions.RequestException as e:
        print(f"❌ Request failed: {e}")

    return None


if __name__ == "__main__":
    if test_connection():
        fetch_logs()
    else:
        print("❌ Cannot fetch logs because connection test failed.")
