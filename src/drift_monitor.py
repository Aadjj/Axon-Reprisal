import logging
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset
from river import drift

class DriftMonitor:
    def __init__(self, threshold=0.7):
        self.threshold = threshold
        self.adwin = drift.ADWIN(delta=0.002)

    def analyze_data_drift(self, reference, current):
        try:
            report = Report(metrics=[DataDriftPreset()])
            report.run(reference_data=reference, current_data=current)
            results = report.as_dict()
            drift_share = results["metrics"][0]["result"]["drift_share"]
            return drift_share > self.threshold
        except Exception as e:
            logging.error(f"Data Drift Analysis Failed: {e}")
            return False

    def detect_concept_drift(self, y_true, y_pred):
        drift_detected = False
        for yt, yp in zip(y_true, y_pred):
            self.adwin.update(int(yt != yp))
            if self.adwin.drift_detected():
                drift_detected = True
        return drift_detected