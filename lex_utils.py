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
                "->":21, # encodes ->
                ".":22,
                "*":23,
                "void":24,
                "return":25,
                "NULL":26,
                "int":27,
                "char":27,
                "double":28,
                "float":29,
                "=":30,

                # var identifiers ( my_int )
                "var0":50,
                "var1":51,
                "var2":52,
                "var3":53,
                "var4":54,
                "var5":55,
                "var6":56,
                "var7":57,
                "var8":58,
                "var9":59,

                # string literals ( "/home/" )
                "str0":60,
                "str1":61,
                "str2":62,
                "str3":63,
                "str4":64,
                "str5":65,
                "str6":66,
                "str7":67,
                "str8":68,
                "str9":69,

                # char literals ( 'a' )
                "char0":70,
                "char1":71,
                "char2":72,
                "char3":73,
                "char4":74,
                "char5":75,
                "char6":76,
                "char7":77,
                "char8":78,
                "char9":79,

                #float literals (123.235)
                "float0":80,
                "float1":81,
                "float2":82,
                "float3":83,
                "float4":84,
                "float5":85,
                "float6":86,
                "float7":87,
                "float8":88,
                "float9":89


                }

    #comments:
    # \ not commonly used outside of strings ( just continues lines )


    ### END Dictionary of Special characters and keywords
    #####################################################

    print(" lexing:")
    i_char = -1

    # literal count
    n_strings = 0
    n_char    = 0
    n_int     = 0
    n_float   = 0

    l_tokens   = []
    l_names    = []
    l_literals = []
    
    n_vars = 0
    d_vars = dict()
    d_str  = dict()
    d_char = dict()
    d_floats = dict()

    this_function+= ' '
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
            st_strg = ""            
            while (this_char != "\""):
                st_strg += this_char
                i_char += 1
                if i_char >= len(this_function): break
                this_char = this_function[i_char]

            if st_strg == "": continue

            if len(st_strg)<10: l_literals.extend([st_strg])
            else: l_literals.extend(["long_str"])
            # check if this str has been used:
            if not (st_strg in d_str):
                d_str[st_strg] = n_strings
                atemp = ("str%d" % n_strings)
                l_names.extend([atemp])
                n_strings +=1
                #add token if str num is in list
                try:
                    l_tokens.extend([cpp_dict[atemp]])
                except:
                    pass
            continue;

        #deal with chars
        if this_char == "\'":            
            i_char += 1
            if i_char >= len(this_function): break
            this_char = this_function[i_char]
            char_string = ""
            while (this_char != "\'"):
                char_string += this_char
                i_char += 1
                if i_char >= len(this_function): break
                this_char = this_function[i_char]

            if char_string == "": continue
            
            l_literals.extend([char_string])
            print("HERE ",char_string)
            # check if this str has been used:
            if not (char_string in d_char):
                d_char[char_string] = n_char
                atemp = ("char%d" % n_char)
                l_names.extend([atemp])
                n_char +=1
                #add token if str num is in list
                try:
                    l_tokens.extend([cpp_dict[atemp]])
                except:
                    pass
            continue;

        # if this_char == "\'":
        #     i_char += 1
        #     if i_char >= len(this_function): break
        #     this_char = this_function[i_char]        
        #     while (this_char != "\'"):
        #         i_char += 1
        #         if i_char >= len(this_function): break
        #         this_char = this_function[i_char]
        #     print("strg%d" % n_strings)
        #     n_strings += 1
        #     continue;
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
                l_literals.extend([astring])
                print("found ",this_char)

            #not a keyword:
            except:
                # check if this variable name has been used before
                if not (astring in d_vars):
                    d_vars[astring] = n_vars
                    l_literals.extend([astring])
                    atemp = ("var%d" % n_vars)                    
                    l_names.extend([atemp])
                    n_vars += 1                    
                    # Add token if var num is in list
                    try:
                        l_tokens.extend([cpp_dict[atemp]])
                    except:
                        pass

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

            # check if number is a float or int
            #float:
            if '.' in nstring:
                l_literals.extend([nstring])

                #check if this float has been used:
                if not (nstring in d_floats):
                    d_floats[nstring] = n_float
                    atemp = ("float%d" % n_float)
                    l_names.extend([atemp])
                    n_float+=1
                    # add token if var num is in list
                    try:
                        l_tokens.extend([cpp_dict[atemp]])
                    #cpp_dict only sees first 10 floats
                    # more floats won't be stored as tokens
                    except:
                        pass
            #int:
            else:
                for subchar in nstring:
                    l_tokens.extend([cpp_dict[subchar]])
                    l_names.extend([subchar])
                    l_literals.extend([subchar])
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
                l_tokens.extend([cpp_dict["->"]])
                l_names.extend(["->"])
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
            l_literals.extend([this_char])
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

    return [l_tokens,l_names,l_literals,
            n_strings+1,n_int+1,n_float+1]
#def lex_func(this_function):
