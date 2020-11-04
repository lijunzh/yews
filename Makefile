#!/bin/sh

.PHONY: install clean test anaconda pypi build wait release

release: test wait clean build wait upload

install:
	pip install -e .

upload:
	$(MAKE) -C anaconda upload
	twine upload dist/* -r pypi

test:
	pytest -m "not large_download" --cov-config .coveragerc --cov yews tests

wait:
	sleep 10

build: anaconda pypi

anaconda:
	$(MAKE) -C $@ build

pypi:
	python setup.py dists
	twine check dist/*

clean:
	find . | grep -E "(__pycache__|\.pyc|\.pyo)" | xargs rm -rf
	$(MAKE) -C anaconda clean
	rm -rf build dist .eggs htmlcov *.npy *.tar.bz2 ._* .coverage .pytest_cache
