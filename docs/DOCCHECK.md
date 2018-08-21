Documentation Checking Process(Only for the developers)
==========================================================  

# Why  

It is necessary for all the developers to generate the rst files which can help us check the documents.  

# When  

1. You add a new function to one of the scripts in the {MatchZoo/matchzoo} or its subdirs  
1. You add a new script to {MatchZoo/matchzoo} or its subdirs  
1. You add a new directory to {MatchZoo/matchzoo} or its subdirs  

# How  
## Make sure you have installed sphinx

1. Enter the docs directory  

```
cd {MatchZoo/docs}
```  

2. Generate the rst files  

```
sphinx-apidoc -f -o source ../matchzoo
```  

3. Commit
