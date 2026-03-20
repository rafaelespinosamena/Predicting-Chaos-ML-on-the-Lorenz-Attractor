.PHONY: install run quick clean lint

install:
	pip install -r requirements.txt

run:
	python main.py

quick:
	python main.py --quick

clean:
	rm -rf figures/ models/ data/ __pycache__ src/__pycache__

lint:
	python -m py_compile main.py
	python -m py_compile src/config.py
	python -m py_compile src/simulate.py
	python -m py_compile src/features.py
	python -m py_compile src/train.py
	python -m py_compile src/visualize.py
	@echo "All modules compile successfully."
