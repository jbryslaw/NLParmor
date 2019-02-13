from lex_utils import lex_func
import sys
import os.path
import numpy as np

import keras
from keras.preprocessing.sequence import pad_sequences
from keras.models import model_from_json

_MAXLEN = 200 #max function length in number of tokens
def predict_single(input_function):
    keras.backend.clear_session()
    # lex the function
    l_tk = lex_func(input_function)

    #load model
    with open('model_rtest_e3.json','r') as model_file:
        js_model = model_file.read()
    model = model_from_json(js_model)
    # load weights
    model.load_weights("model_rtest_e3.h5")
    model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['acc'])
    print(model.summary())

    X_matrix = np.array([l_tk[0]])
    X_padded = pad_sequences(X_matrix,maxlen=_MAXLEN,truncating="post", padding="post",value=265)

    predict = model.predict(X_padded)

    print(" Prob Found: ",predict[:,1])

    return predict[:,1]
    
def main():
    input_function = "a"

    # read function from file if one is passed
    if len(sys.argv) == 2:
        txt_infile = sys.argv[1]
        if(os.path.isfile(txt_infile)):
            with open(txt_infile,'r') as thisfile:
                input_function = thisfile.read()

    predict_single(input_function)

if __name__ == "__main__":
    main()

