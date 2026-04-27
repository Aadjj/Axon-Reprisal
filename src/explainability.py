import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import logging
from lime import lime_tabular
import os


class XAIEngine:
    def __init__(self, model, training_data):
        self.model = model
        self.data = training_data

        os.makedirs("logs", exist_ok=True)

        self.explainer = lime_tabular.LimeTabularExplainer(
            training_data=self.data.values,
            feature_names=self.data.columns.tolist(),
            class_names=['Normal', 'Threat'],
            mode='regression'
        )

    def explain_instance(self, instance_index):
        try:
            instance = self.data.iloc[instance_index].values

            def predict_fn(x):
                preds = self.model.predict(x, verbose=0)
                return np.mean(np.abs(preds - x), axis=1) if len(preds.shape) > 1 else preds

            exp = self.explainer.explain_instance(
                data_row=instance,
                predict_fn=predict_fn,
                num_features=5
            )

            fig = exp.as_pyplot_figure()
            plt.title(f"LIME Explanation: Instance {instance_index}")
            plt.tight_layout()
            plt.savefig(f"logs/anomaly_explanation_{instance_index}.png")
            plt.close(fig)

            reasons = exp.as_list()
            logging.info(f"Anomaly Reason Codes for Index {instance_index}: {reasons}")

            return reasons

        except Exception as e:
            logging.error(f"LIME Instance Explanation Failed: {e}")
            return None

    def get_global_importance(self):
        try:
            logging.info("Generating Global Feature Importance via LIME aggregation...")

            sample_indices = np.random.choice(self.data.index, min(20, len(self.data)), replace=False)
            global_weights = {feat: 0 for feat in self.data.columns}

            def predict_fn(x):
                return np.mean(np.abs(self.model.predict(x, verbose=0) - x), axis=1)

            for idx in sample_indices:
                exp = self.explainer.explain_instance(self.data.iloc[idx].values, predict_fn)
                for feat, weight in exp.as_list():
                    for real_feat in self.data.columns:
                        if real_feat in feat:
                            global_weights[real_feat] += abs(weight)

            plt.figure(figsize=(12, 8))
            sorted_weights = dict(sorted(global_weights.items(), key=lambda item: item[1], reverse=True))
            plt.barh(list(sorted_weights.keys()), list(sorted_weights.values()), color='#4a90e2')
            plt.title("AXON-REPRISAL: Global Feature Influence")
            plt.xlabel("Aggregated LIME Weight")
            plt.gca().invert_yaxis()
            plt.savefig("logs/global_feature_importance.png")
            plt.close()

        except Exception as e:
            logging.error(f"Global LIME Analysis Failed: {e}")