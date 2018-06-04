init:
		pip install -r requirements.txt --user

test:
		pytest --doctest-modules --doctest-continue-on-failure --cov matchzoo/ --cov-report term-missing --cov-config .coveragerc matchzoo/ tests/
		flake8 ./matchzoo --exclude __init__.py
