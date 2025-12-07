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

test:
	pytest -q

