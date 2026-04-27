import os
import logging
import pandas as pd
import numpy as np
from dotenv import load_dotenv
from stable_baselines3 import PPO
from src.preprocessing import load_and_preprocess_data
from src.detection import DeepAxonEngine
from src.graph_engine import detect_lateral_movement, build_network_graph
from src.explainability import XAIEngine
from src.response_env import ThreatResponseEnv
from src.drift_monitor import DriftMonitor
from src.intel_bridge import fetch_threat_intel

load_dotenv()

def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    logging.info("--- AXON-REPRISAL v3.0: SOVEREIGN DEFENSE ONLINE ---")

    feature_cols = [
        "cpu_usage", "memory_usage", "network_traffic_in", "network_traffic_out",
        "user_activity", "disk_io", "clock_skew", "tcp_window_size", "io_entropy",
        "cpu_usage_velocity", "network_traffic_in_velocity"
    ]
    raw_log_path = "data/raw/system_logs.csv"

    data = load_and_preprocess_data(raw_log_path, feature_cols)
    if data is None:
        logging.error("CRITICAL: Data pipeline failure. Shutdown initiated.")
        return

    engine = DeepAxonEngine(sequence_length=10, feature_count=len(data.columns))
    logging.info("Training Neural Reconstruction Engine...")
    engine.train_model(data)
    detection_results = engine.run_inference(data)

    logging.info("Building Network Relationship Graph (GNN)...")
    edge_list = build_network_graph(data)
    lateral_threats = detect_lateral_movement(data.values, edge_list)

    anomalies = data[detection_results["prediction"] == -1]
    total_threats = len(anomalies) + lateral_threats.sum().item()
    logging.info(f"ALERTS: {len(anomalies)} Temporal Anomalies | {lateral_threats.sum().item()} Lateral Movements.")

    if total_threats > 0:
        xai = XAIEngine(engine.model, data)
        env = ThreatResponseEnv()

        try:
            rl_model = PPO.load("models/battle_hardened_axon", env=env)
            logging.info("Autonomous Sovereign Agent Online (Adversarial PPO).")
        except Exception:
            rl_model = None
            logging.warning("Sovereign Agent Offline. Defaulting to manual analysis.")

        for idx in anomalies.index[:5]:
            logging.info(f"--- INVESTIGATING THREAT {idx} ---")

            reasons = xai.explain_instance(idx)

            intel = fetch_threat_intel(
                "8.8.8.8",
                os.getenv("THREAT_INTEL_URL"),
                os.getenv("THREAT_INTEL_API_KEY")
            )

            if rl_model:
                state = env.reset(initial_state=data.iloc[idx].values)
                action, _ = rl_model.predict(state, deterministic=True)
                _, reward, _, info = env.step(action)
                logging.info(f"[ACTION] {info['action_taken']} | System Integrity Reward: {reward}")

    monitor = DriftMonitor()
    if monitor.analyze_data_drift(data, data):
        logging.warning("SYSTEM DRIFT: Environment has shifted. Retraining scheduled.")

    logging.info("--- SCAN COMPLETE: AXON-REPRISAL STANDING BY ---")

if __name__ == "__main__":
    main()