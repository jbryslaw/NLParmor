def lex_func(this_function):
    #####################################################
    ####### Dictionary of Special characters and keywords
    cpp_dict = {"0":0,
                "1":1,
                "2":2,
                "3":3,
                "4":4,
                "5":5,
                "6":6,
                "7":7,
                "8":8,
                "9":9,
                "(":10,
                ")":11,
                "[":12,
                "]":13,
                "{":14,
                "}":15,
                "|":16,            
                ":":17,
                ";":18,
                "~":19,
                "!":20,
                "arrow":21, # encodes ->
                ".":22,
                "*":23,
                "void":24,
                "return":25,
                "NULL":26,
                "int":27,
                "double":28,
                "float":29,
                "=":30
                }

    #comments:
    # \ not commonly used outside of strings ( just continues lines )


    ### END Dictionary of Special characters and keywords
    #####################################################

    print(" lexing:")
    i_char = -1

    # literal count
    n_strings = 0
    n_int     = 0
    n_float   = 0

    l_tokens = []
    l_names  = []

    n_vars = 0
    d_vars = dict()
    while i_char < (len(this_function)-1):
        i_char +=1
        if i_char >= len(this_function): break
        this_char = this_function[i_char]

        #ignore whitespace
        if this_char.isspace(): continue

        print(" lexing c: ",this_char)
        ##############################
        # clear comments
        if this_char == "/" and (i_char+1 < len(this_function)):
            if this_function[i_char+1] == "/":
                while (this_char != "\n") and (this_char != "\r"):
                    i_char += 1
                    if i_char >= len(this_function): break
                    this_char = this_function[i_char]
            elif this_function[i_char+1] == "*":
                i_char += 1
                if i_char >= len(this_function): break
                while (this_char != "*") and (this_function[i_char+1] != "/"):                
                    i_char += 1
                    if i_char >= len(this_function): break
                    this_char = this_function[i_char]
                pass
        # END clear comments
        ##############################
        
        ##############################
        #deal with strings
        if this_char == "\"":
            i_char += 1
            if i_char >= len(this_function): break
            this_char = this_function[i_char]        
            while (this_char != "\""):
                i_char += 1
                if i_char >= len(this_function): break
                this_char = this_function[i_char]
            print("strg%d" % n_strings)
            n_strings += 1
            i_char-=1
            continue;

        if this_char == "\'":
            i_char += 1
            if i_char >= len(this_function): break
            this_char = this_function[i_char]        
            while (this_char != "\'"):
                i_char += 1
                if i_char >= len(this_function): break
                this_char = this_function[i_char]
            print("strg%d" % n_strings)
            n_strings += 1
            i_char-=1
            continue;
        # END deal with strings
        ##############################
        
        ##############################
        #check for alphanumeric string
        if this_char.isalpha() or this_char == '_':
            astring = this_char
            i_char += 1
            if i_char >= len(this_function): break
            this_char = this_function[i_char]
            while this_char.isalnum() or this_char == '_':
                astring += this_char
                i_char += 1
                if i_char >= len(this_function): break
                this_char = this_function[i_char]

            print(astring)
            #  ########check alphanumeric string for keywords, and tokenize
            try:
                l_tokens.extend([cpp_dict[astring]])
                l_names.extend([astring])
                print("found ",this_char)
            except:
                if not (astring in d_vars):
                    d_vars[astring] = n_vars
                    astring = ("var%d" % n_vars)                    
                    l_names.extend([astring])
                    print("this: ",astring)
                    n_vars += 1
                    ### need to add toke code 

            i_char-=1
            continue;

        # END check for alphanumeric string
        ##############################



        ##############################
        # check for numbers
        if this_char.isdigit() or this_char == '.':
            nstring = this_char
            i_char += 1
            if i_char >= len(this_function): break
            this_char = this_function[i_char]
            while this_char.isdigit() or this_char == '.':
                nstring += this_char
                i_char += 1
                if i_char >= len(this_function): break
                this_char = this_function[i_char]
            print(nstring)
            l_names.extend([nstring])
            
            i_char-=1
            continue;
        # check for numbers
        ##############################

        # ############# Tokenize numbers

        ##############################
        # check for arrows
        if (i_char+1) < len(this_function):
            if (this_char == "-") and this_function[i_char+1] == ">":
                i_char+=2
                if i_char >= len(this_function): break
                this_char = this_function[i_char]
                l_tokens.extend([cpp_dict["arrow"]])
                l_names.extend(["->"])
                print("found arrow skipping to character: ", i_char," ",this_char)
                i_char-=1
                continue;
        # END check for arrows
        ##############################

        ##need to check other digraphs and trigraphs

        #IGNORE SEMICOLONS????
        if this_char == ";": continue            

        ##############################
        ######## Tokenize special characters
        # if this_char == "]":        
        #     l_tokens.extend([cpp_dict[this_char]])
        try:
            l_tokens.extend([cpp_dict[this_char]])
            l_names.extend([this_char])
            print("found ",this_char)
        except:
            print("\"",this_char,"\" not in dictionary")
        #### END Tokenize special characters
        ##############################

        #print(this_char)




        # deal with &d, %d etc

        # deal with __ __

    #while i_char < (len(this_function)-1):

    # return a list with the tokens and their identifiers

    print("keys: ",d_vars.keys())
    return [l_tokens,l_names,n_strings,n_int,n_float]
#def lex_func(this_function):
