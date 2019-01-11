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
sphinx-apidoc -o source/ ../matchzoo/
# Generate html from rst
make clean
make html
```

