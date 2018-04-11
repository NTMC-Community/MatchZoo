init:
	pip install -r requirements.txt

test:
	pytest tests/  --cov matchzoo/ --cov-report term-missing --cov-config .coveragerc
	flake8 ./matchzoo --exclude __init__.py
