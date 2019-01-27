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

                #single characters
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
                ".":21,
                "*":22,
                "=":23,
                "+":24,
                "-":25,
                "/":26,
                "%":27,
                "<":28,
                ">":29,
                "&":30,
                "^":31,
                "?":32,
                ",":33,
                "#":34,

                #digraphs
                "->":35,
                ">>":36,
                "<<":37,
                "::":38,
                "<=":39,
                ">=":40,
                "++":41,
                "--":42,
                "==":43,
                "!=":44,
                "&&":45,
                "||":46,
                "?:":47,
                "+=":48,
                "-=":49,
                "/=":50,
                "&=":51,
                "|=":52,
                "^=":53,
                "%=":54,
                ".*":55,
                "*=":56,
                "##":57,
                "<:":58,
                ":>":59,
                "<%":60,
                "%>":61,
                "%:":62,
                                
                #trigraphs
                "<<=":63,
                ">>=":64,
                "??>":65,
                "??<":66,
                "??(":67,
                "??(":68,
                "??=":69,
                "??/":70,
                "??'":71,
                "??!":72,
                "??-":73,
                
                #keywords
                "alignas":74,
                "alignof":75,
                "and":76,
                "and_eq":77,
                "asm":78,
                "atomic_cancel":79,
                "atomic_commit":80,
                "atomic_noexcept":81,
                "auto":82,
                "bitand":83,
                "bitor":84,
                "bool":85,
                "break":86,
                "case":87,
                "catch":88,
                "char":89,
                "char8_t":90,
                "char16_t":91,
                "char32_t":92,
                "class":93,
                "compl":94,
                "concept":95,
                "const":96,
                "consteval":97,
                "constexpr":98,
                "const_cast":99,
                "continue":100,
                "co_await":101,
                "co_return":102,
                "co_yield":103,
                "decltype":104,
                "default":105,
                "delete":106,
                "do":107,
                "double":108,
                "dynamic_cast":109,
                "else":110,
                "enum":111,
                "explicit":112,
                "export":113,
                "extern":114,
                "false":115,
                "float":116,
                "for":117,
                "friend":118,
                "goto":119,
                "if":120,
                "import":121,
                "inline":122,
                "int":123,
                "long":124,
                "module":125,
                "mutable":126,
                "namespace":127,
                "new":128,
                "noexcept":129,
                "not":130,
                "not_eq":131,
                "nullptr":132,
                "operator":133,
                "or":134,
                "or_eq":135,
                "private":136,
                "protected":137,
                "public":138,
                "reflexpr":139,
                "register":140,
                "reinterpret_cast":141,
                "requires":142,
                "return":143,
                "short":144,
                "signed":145,
                "sizeof":146,
                "static":147,
                "static_assert":148,
                "static_cast":149,
                "struct":150,
                "switch":151,
                "synchronized":152,
                "template":153,
                "this":154,
                "thread_local":155,
                "throw":156,
                "true":157,
                "try":158,
                "typedef":159,
                "typeid":160,
                "typename":161,
                "union":162,
                "unsigned":163,
                "using":164,
                "virtual":165,
                "void":166,
                "volatile":167,
                "wchar_t":168,
                "while":169,
                "xor":170,
                "xor_eq":171,
                "override":172,
                "final":173,
                "audit":174,
                "axiom":175,
                "transaction_safe":176,
                "transaction_safe_dynamic":177,

                "if":178,
                "elif":179,
                "else":180,
                "endif":181,
                "ifdef":182,
                "ifndef":183,
                "define":184,
                "undef":185,
                "include":186,
                "line":187,
                "error":188,
                "pragma":189,
                "defined":190,
                "__has_include":191,
                "__has_cpp_attribute":192,
                "_Pragma":193,

                # var identifiers ( my_int )
                "var0":194,
                "var1":195,
                "var2":196,
                "var3":197,
                "var4":198,
                "var5":199,
                "var6":200,
                "var7":201,
                "var8":202,
                "var9":203,

                "var10":204,
                "var11":205,
                "var12":206,
                "var13":207,
                "var14":208,
                "var15":209,
                "var16":210,
                "var17":211,
                "var18":212,
                "var19":213,

                "var20":224,
                "var21":225,
                "var22":226,
                "var23":227,
                "var24":228,
                "var25":229,
                "var26":220,
                "var27":221,
                "var28":222,
                "var29":223,


                # string literals ( "/home/" )
                "str0":234,
                "str1":235,
                "str2":236,
                "str3":237,
                "str4":238,
                "str5":239,
                "str6":240,
                "str7":241,
                "str8":242,
                "str9":243,

                # char literals ( 'a' )
                "char0":244,
                "char1":245,
                "char2":246,
                "char3":247,
                "char4":248,
                "char5":249,
                "char6":250,
                "char7":251,
                "char8":252,
                "char9":253,

                #float literals (123.235)
                "float0":254,
                "float1":255,
                "float2":256,
                "float3":257,
                "float4":258,
                "float5":259,
                "float6":260,
                "float7":261,
                "float8":262,
                "float9":263                
                }

    #comments:
    # \ not commonly used outside of strings ( just continues lines )


    ### END Dictionary of Special characters and keywords
    #####################################################

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
        # check for trigraphs
        #if this and next char and next are dict ...
        if (i_char+2) < len(this_function):
            tri_strg = this_char
            tri_strg += this_function[i_char+1]
            tri_strg += this_function[i_char+2]
            try:
                l_tokens.extend([cpp_dict[tri_strg]])
                l_names.extend([tri_strg])
                l_literals.extend([tri_strg])
                i_char+=3
                continue
            except:
                pass
        # END check for trigraphs
        ##############################

        ##############################
        # check for digraphs
        #if this and next char are dict ...
        if (i_char+1) < len(this_function):
            di_strg = this_char
            di_strg += this_function[i_char+1]
                    
            try:
                l_tokens.extend([cpp_dict[di_strg]])
                l_names.extend([di_strg])
                l_literals.extend([di_strg])
                i_char+=2
                continue
            except:
                pass
        # END check for digraphs
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
            i_strings = 0
            if not (st_strg in d_str):
                d_str[st_strg] = n_strings
                i_strings = n_strings
                n_strings +=1
            else: i_strings = d_str[st_strg]
                
            atemp = ("str%d" % i_strings)
            l_names.extend([atemp])
                
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

            #  ########check alphanumeric string for keywords, and tokenize
            try:
                l_tokens.extend([cpp_dict[astring]])
                l_names.extend([astring])
                l_literals.extend([astring])

            #not a keyword:
            except:
                # check if this variable name has been used before
                i_vars = 0
                if not (astring in d_vars):
                    d_vars[astring] = n_vars
                    i_vars = n_vars
                    n_vars += 1
                else:
                    i_vars = d_vars[astring]

                atemp = ("var%d" % i_vars)
                # right now, writing duplicates to tokenization
                l_literals.extend([astring])                    
                l_names.extend([atemp])
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
        #make sure its not just a dot
        b_justdot = False
        if this_char == '.':
            if i_char >= len(this_function): break
            if not this_function[i_char+1].isdigit(): b_justdot = True
        
        if ( not b_justdot ) and (this_char.isdigit() or this_char == '.'):
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

        ##need to check other digraphs and trigraphs

        #IGNORE SEMICOLONS????
        if this_char == ";": continue            

        ##############################
        ######## Tokenize special characters
        try:
            l_tokens.extend([cpp_dict[this_char]])
            l_names.extend([this_char])
            l_literals.extend([this_char])
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
