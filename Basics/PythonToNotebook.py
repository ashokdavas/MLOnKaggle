import IPython.nbformat.current as nbf
nb = nbf.read(open('IrisAnalysis.py', 'r'), 'py')
nbf.write(nb, open('test.ipynb', 'w'), 'ipynb')