.PHONY: install train experiment dashboard test clean

install:
	pip install -r requirements.txt

train:
	python -m src.train

experiment:
	python -c "from src.utils.logging import setup_logging; from src.experiments import run_all_experiments; setup_logging(); run_all_experiments(n_episodes=500, n_seeds=3)"

dashboard:
	PYTHONPATH=. streamlit run dashboard/app.py

test:
	pytest tests/ -v --cov=src --cov-report=term-missing

clean:
	rm -rf experiments/logs/* experiments/checkpoints/*
	find . -type d -name __pycache__ -exec rm -rf {} +