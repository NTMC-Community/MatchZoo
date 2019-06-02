## Build Documentation:



#### Install Requirements

```python
pip install -r requirements.txt
```



#### Build Documentation

```python
# Enter docs folder.
cd docs
# Use sphinx autodoc to generate rst.
# usage: sphinx-apidoc [OPTIONS] -o <OUTPUT_PATH> <MODULE_PATH> [EXCLUDE_PATTERN,...]
sphinx-apidoc -o source/ ../matchzoo/ ../matchzoo/contrib
# Generate html from rst
make clean
make html
```

