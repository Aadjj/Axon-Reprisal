setup:
	pip install -r requirements.txt
	mkdir -p data/raw data/processed data/external models logs reports

data:
	python generate_mock_data.py

train:
	python train_rl.py

run:
	python main.py

gui:
	python ui_dashboard.py

all:
	python generate_mock_data.py
	python train_rl.py
	python main.py
	python ui_dashboard.py

clean:
	rm -rf logs/*.png
	rm -rf reports/*.html
	rm -rf models/*.joblib
	rm -rf models/*.zip