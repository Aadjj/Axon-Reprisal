import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os


def generate_data():
    os.makedirs("data/raw", exist_ok=True)
    os.makedirs("data/external", exist_ok=True)

    rows =1000
    start_time = datetime.now() - timedelta(days=1)

    data = {
        "timestamp": [start_time + timedelta(minutes=i) for i in range(rows)],
        "cpu_usage": np.random.uniform(10, 40, rows),
        "memory_usage": np.random.uniform(20, 50, rows),
        "network_traffic_in": np.random.uniform(100, 500, rows),
        "network_traffic_out": np.random.uniform(100, 500, rows),
        "user_activity": np.random.randint(1, 10, rows),
        "disk_io": np.random.uniform(5, 20, rows)
    }

    df = pd.DataFrame(data)

    for i in range(10):
        idx = np.random.randint(0, rows)
        df.loc[idx, "cpu_usage"] = 95.0
        df.loc[idx, "network_traffic_out"] = 5000.0

    df.to_csv("data/raw/system_logs.csv", index=False)

    labels = pd.DataFrame({
        "index": range(rows),
        "threat_type": np.random.choice(["Normal", "DDoS", "Exfiltration"], rows, p=[0.98, 0.01, 0.01])
    })
    labels.to_csv("data/external/threat_labels.csv", index=False)
    print("Mock enterprise data generated successfully.")


if __name__ == "__main__":
    generate_data()