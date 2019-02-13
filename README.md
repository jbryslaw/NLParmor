# NLP armor

Software can contain 0-Day or hidden vulnerabilities which may allow malicious actors to gain access to secure areas in server memory. Some of the recent mega-breaches can be traced back to such vulnerabilities. Static analyzers, although easy to use, miss up to 60% of hidden vulnerabilities. Dynamic analyzers have a better detection rate, but are much more time consuming and require experts to run.

NLP Armor, utilizes strategies from natural language processing to quickly and effectively scan C++ functions for vulnerabilities. A custom python package, lex_utils, tokenizes C++ functions. These tokens are then inputted into an embedding layer in a convolution neural network. The CNN was trained on set of C++ functions labeled according to vulnerability. The trained CNN can then detect vulnerabilities in C++ code inputted by the user. NLP armor provides a better detection rate than static analyzers and is easier to use and less time consuming than dynamic analyzers.

[Slides](https://drive.google.com/open?id=1HGSWpMnimTQN8Xq9kPlOag85D1KDAqmqaD3IfkittSo)

# Code Description
1. lex_utils.py:
   Python package for tokenizing C++;
2. read_to_lex.py:
   Tests the tokenizer, taking C++ source files as input.
3. NLParmor_train.py:
   Trains the convolution neural network.
4. NLParmor_model.py:
   Keras implementation of a convultion neural network for learning source code embeddings.
5. NLParmor_single.py:
   Takes a single C++ source file as input. Uses the CNN weights trained by NLParmor_model.py, to predict the probability that the inputted file contains a known vulnerability.
