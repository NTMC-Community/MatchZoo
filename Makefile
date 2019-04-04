init:
	pip install -r requirements.txt

TEST_ARGS = --doctest-modules --doctest-continue-on-failure --cov matchzoo/ --cov-report term-missing --cov-report html --cov-config .coveragerc matchzoo/ tests/ -W ignore::DeprecationWarning
FLAKE_ARGS = ./matchzoo --exclude=__init__.py,matchzoo/contrib

test:
	pytest $(TEST_ARGS)
	flake8 $(FLAKE_ARGS)

quick:
	pytest -m 'not slow' $(TEST_ARGS)

slow:
	pytest -m 'slow' $(TEST_ARGS)

flake:
	flake8 $(FLAKE_ARGS)
