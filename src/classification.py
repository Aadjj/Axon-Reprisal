import os
import joblib
import logging
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report


class ThreatClassifier:
    def __init__(self, model_dir="models", random_state=42):
        self.model_dir = model_dir
        self.random_state = random_state
        self.model_path = os.path.join(self.model_dir, "threat_classifier.joblib")

    def train(self, X, y):
        try:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=self.random_state, stratify=y
            )
            model = GradientBoostingClassifier(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=5,
                random_state=self.random_state
            )
            model.fit(X_train, y_train)

            joblib.dump(model, self.model_path)

            predictions = model.predict(X_test)
            logging.info(f"Classification Report:\n{classification_report(y_test, predictions)}")
            return model
        except Exception as e:
            logging.error(f"Classification Training Failed: {e}")
            return None

    def predict(self, model, data):
        try:
            return model.predict(data)
        except Exception as e:
            logging.error(f"Classification Prediction Failed: {e}")
            return None
