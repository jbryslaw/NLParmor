from lex_utils import lex_func
import sys
import os.path
import numpy as np

# read function from file if one is passed
this_function = ""
if len(sys.argv) == 2:
    txt_infile = sys.argv[1]
    if(os.path.isfile(txt_infile)):
        with open(txt_infile,'r') as thisfile:
            this_function = thisfile.read()
    
# lex the function
l_tk = lex_func(this_function)

#print("Tokens:")
print(l_tk[0])
#print("Name:")
print()
print(l_tk[1])
#print("Literals:")
print()
print(l_tk[2])
