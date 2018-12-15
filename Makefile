init:
	pip install -r requirements.txt

test:
	pytest --doctest-modules --doctest-continue-on-failure --cov matchzoo/ --cov-report term-missing --cov-config .coveragerc matchzoo/ tests/
	flake8 ./matchzoo --exclude __init__.py

quick:
	pytest -m 'not slow' --doctest-modules --doctest-continue-on-failure --cov matchzoo/ --cov-report term-missing --cov-config .coveragerc matchzoo/ tests/unit_test

slow:
	pytest -m 'slow' --doctest-modules --doctest-continue-on-failure --cov matchzoo/ --cov-report term-missing --cov-config .coveragerc matchzoo/ tests/unit_test

flake:
	flake8 ./matchzoo --exclude __init__.py
