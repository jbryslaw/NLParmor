import keras
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.models import model_from_json
from keras.models import Model
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

from NLParmor_model import NLP_model

#################################
### Consts
_MAXLEN = 200 #max function length in number of tokens
#_MAXLEN = 400 #max function length in number of tokens 
_MAXLEN = _MAXLEN
FRAC_TEST=.2 #Fraction of data to set aside for testing
TOKEN_UNIVERSE = 266 # Number of unique tokens
N_epoch = 12
#################################

#################################
### switches
b_plot      = False #True #False
b_load_model_from_file = True #False #True #False # True #False #True #False #True #False
#################################

#################################
### training file
df_total  = pd.read_pickle("train0_rebalenced.plk")
#################################

################################
## Plot 
if b_plot:
    t_lens = df_total.iloc[:,6].str.len()
    fig = plt.figure(figsize=(10, 10))
    ax = plt.subplot(111)
    ax.set_yscale("log")
    plt.xlabel('Sentence length')
    plt.ylabel('Number of sentences')
    plt.hist(t_lens, bins=40)
    plt.show()
###############################

###############################
# Pad sequences with to Max Length
#   and Truncate at max length
# pad w/ value 265, since integer tokens range from 0 to 266
#need to reset the index
df_total.index =range(len(df_total))

def pad_length(this_row,MAXLEN = 10):
    a_padded = pad_sequences([this_row],maxlen=MAXLEN,truncating="post", padding="post",value=265)

    return a_padded[0]
l_seq = [f's{i}' for i in range(0,_MAXLEN)]
df_total[l_seq] = pd.DataFrame(pad_sequences(df_total.Tokens,maxlen=_MAXLEN,truncating="post", padding="post",value=265), columns = l_seq)
### END Padding
###############################

#convert df to arrays
padded_tokens = df_total[l_seq].values
labels = to_categorical(np.asarray(df_total['any-vuln']))

# get number of test rows
N_test = int(FRAC_TEST * padded_tokens.shape[0])

# initialze embedding weights to random
np.random.seed(1)
w_embed = np.random.rand(TOKEN_UNIVERSE, TOKEN_UNIVERSE)

train_X = padded_tokens[:-N_test]
train_y = labels[:-N_test]
test_X  = padded_tokens[-N_test:]
test_y  = labels[-N_test:]

# Train new model, or load from file
if not b_load_model_from_file:
    model = NLP_model(w_embed, _MAXLEN, TOKEN_UNIVERSE, TOKEN_UNIVERSE, 2)
    model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['acc'])
    print(model.summary())
    model.fit(train_X, train_y, epochs=N_epoch, batch_size=128, validation_data=(test_X, test_y))
else:
    #load model
    with open('model_rtest_e3.json','r') as model_file:
        js_model = model_file.read()
        model = model_from_json(js_model)
        # load weights
    model.load_weights("model_rtest_e3.h5")
    model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['acc'])
    print(model.summary())
    #model.compile(loss='binary_crossentropy', optimizer='rmsprop',metrics=['accuracy'])
    
predict = model.predict(test_X)
y_pred   = predict[:,1] > 0.5
y_1D_test = test_y[:,1] == 1 
cm = confusion_matrix(y_1D_test,y_pred)
if len(y_1D_test) != 0.: cm = cm/len(y_1D_test)
print(cm)

precision = cm[0,0] / (cm[0,0]+cm[0,1])
print(" p: ",precision)
recall    = cm[0,0] / (cm[0,0]+cm[1,0])
print(" r: ",recall)

##############################
### save model to file
if not b_load_model_from_file:
    # serialize model to JSON
    model_json = model.to_json()
    c_js_out = "model_rtest_e3.json"
    c_w_out  = "model_rtest_e3.h5"
    with open(c_js_out, "w") as json_file:
        json_file.write(model_json)
        # serialize weights to HDF5
    model.save_weights(c_w_out)
    print("Saved model to disk")
##############################
##############################
