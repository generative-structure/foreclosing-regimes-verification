PY ?= python3

.PHONY: gate checks analyse simulate

gate:
	$(PY) sim/gate.py

checks:
	@for f in checks/*.py; do echo "== $$f =="; $(PY) "$$f" || exit 1; done

analyse:
	$(PY) sim/analyse.py

simulate:
	$(PY) sim/run.py
