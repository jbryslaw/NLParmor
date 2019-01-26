from lex_utils import lex_func
import sys
import os.path

this_function = " a->"
# read function from file if one is passed
if len(sys.argv) == 2:
    txt_infile = sys.argv[1]
    if(os.path.isfile(txt_infile)):
        with open(txt_infile,'r') as thisfile:
            this_function = thisfile.read()
    
print(this_function)


# lex the function
l_tk_names = lex_func(this_function)

    
#while i_char < (len(this_function)-1):
print(" tokens found:")
print(l_tk_names[1])

    
print("Number of literals:")
print(" N Strgs: ",l_tk_names[2])
print(" N int: ",l_tk_names[3])
print(" N float: ",l_tk_names[4])
print()
print("Function:")
print(this_function)
print("Tokens:")
print(l_tk_names[1])
print()
