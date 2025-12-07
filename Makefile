PYTHON=python3

setup:
	$(PYTHON) -m venv venv
	./venv/bin/pip install -r requirements.txt

phase1a:
	$(PYTHON) -m ingestion.nfl.run_phase1a

phase1c:
	$(PYTHON) -m orchestration.pipelines.phase1c_pipeline

phase1d:
	$(PYTHON) -m orchestration.pipelines.phase1d_pipeline

phase2:
	$(PYTHON) -m orchestration.pipelines.phase2_pipeline

phase2b:
	$(PYTHON) -m orchestration.pipelines.phase2b_pipeline

sample:
	$(PYTHON) -m orchestration.pipelines.phase1_sample_pipeline

test:
	pytest -q

test-sample:
	pytest tests/test_phase1_sample_pipeline.py -v

clean:
	rm -rf data/nfl/sample/processed/*
	rm -rf docs/reports/sample_phase1.md
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true

.PHONY: setup phase1a phase1c phase1d phase2 phase2b sample test test-sample clean

