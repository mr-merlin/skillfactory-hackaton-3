# Makefile for Perfumery Organ Recommender
# on Windows use: nmake or run commands manually

PYTHON  = .venv/Scripts/python
PIP     = .venv/Scripts/pip
DATA    = data

.PHONY: install evaluate api demo analytics notebook lint help

## install  - создать venv и установить зависимости
install:
	python -m venv .venv
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt

## evaluate - запустить полную оценку метрик (Hit@K, MRR, NDCG, Diversity, Stability, Latency)
evaluate:
	$(PYTHON) evaluate.py --data-dir $(DATA)

## api      - запустить REST API (http://localhost:8000)
api:
	$(PYTHON) -m uvicorn src.api:app --host 0.0.0.0 --port 8000 --reload

## demo     - консольная демонстрация рекомендаций для сессии 1
demo:
	$(PYTHON) -m src.cli recommend-session --session-id 1 --top-n 10 --data-dir $(DATA)

## analytics - сформировать аналитический отчёт (analytics_report.md)
analytics:
	$(PYTHON) analytics.py --data-dir $(DATA)

## notebook  - запустить Jupyter Lab с EDA ноутбуком
notebook:
	$(PYTHON) -m jupyter lab notebooks/eda.ipynb

## lint      - проверка кода (flake8)
lint:
	$(PYTHON) -m flake8 src/ evaluate.py analytics.py --max-line-length 120 --ignore=E501,W503

## help      - справка по командам
help:
	@echo ""
	@echo "Доступные команды:"
	@echo "  make install    - установить зависимости"
	@echo "  make evaluate   - запустить оценку метрик"
	@echo "  make api        - запустить REST API"
	@echo "  make demo       - демонстрация CLI"
	@echo "  make analytics  - аналитический отчёт"
	@echo "  make notebook   - Jupyter Lab"
	@echo "  make lint       - проверка кода"
	@echo ""
