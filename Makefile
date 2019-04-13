# Usages:
#
# to install matchzoo dependencies:
# $ make init
#
# to run all matchzoo tests, recommended for big PRs and new versions:
# $	make test
#
# there are three kinds of tests:
#
# 1. "quick" tests
# 	- run in seconds
#   - include all unit tests without marks and all doctests
#   - for rapid prototyping
#   - CI run this for all PRs
#
# 2. "slow" tests
#   - run in minutes
#   - include all unit tests marked "slow"
#   - CI run this for all PRs
#
# 3. "cron" tests
#   - run in minutes
#   - involves underministic behavoirs (e.g. network connection)
#   - include all unit tests marked "cron"
#   - CI run this on a daily basis
#
# to run quick tests, excluding time consuming tests and crons:
# $ make quick
#
# to run slow tests, excluding normal tests and crons:
# $ make slow
#
# to run crons:
# $ make cron
#
# to run all tests:
# $ make test
#
# to run CI push/PR tests:
# $ make push
#
# to run docstring style check:
# $ make flake

init:
	pip install -r requirements.txt

TEST_ARGS = -v --full-trace -l --doctest-modules --doctest-continue-on-failure --cov matchzoo/ --cov-report term-missing --cov-report html --cov-config .coveragerc matchzoo/ tests/ -W ignore::DeprecationWarning --ignore=matchzoo/contrib
FLAKE_ARGS = ./matchzoo --exclude=__init__.py,matchzoo/contrib

test:
	pytest $(TEST_ARGS)
	flake8 $(FLAKE_ARGS)

push:
	pytest -m 'not cron' $(TEST_ARGS) ${ARGS}
	flake8 $(FLAKE_ARGS)

quick:
	pytest -m 'not slow and not cron' $(TEST_ARGS) ${ARGS}

slow:
	pytest -m 'slow and not cron' $(TEST_ARGS) ${ARGS}

cron:
	pytest -m 'cron' $(TEST_ARGS) ${ARGS}

flake:
	flake8 $(FLAKE_ARGS) ${ARGS}
