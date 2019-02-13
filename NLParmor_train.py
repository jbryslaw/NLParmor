import keras
import keras.backend as kb
import tensorflow as tf
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.models import model_from_json
from keras.models import Model
from keras.utils import plot_model
from keras.callbacks import ModelCheckpoint
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.metrics import matthews_corrcoef
from sklearn.metrics import f1_score
#import plot_confusion_matrix
from NLParmor_model import NLP_model
import itertools
import losses
from plt_histogram import plt_histogram

#################################
### Consts
_MAXLEN = 200 #max function length in number of tokens
#_MAXLEN = 100 #max function length in number of tokens
#_MAXLEN = 400 #max function length in number of tokens 
_MAXLEN = _MAXLEN
FRAC_TEST=.2 #Fraction of data to set aside for testing
TOKEN_UNIVERSE = 266 # Number of unique tokens
N_epoch = 300
# there are 29 functions without vulnerabilites
#    to every 2 vulnerable functions
class_weights = {0:2,1:29}
#################################

#################################
### switches
b_plot       = False#True 
b_load_model_from_file = False #True 
# do a binary classification (yes/no vuln) or categorical classification
b_binary     = True #False
b_start_from_previous_model = True
b_save_model = True #False
#################################

#################################
### training file
df_total  = pd.read_pickle("all_train.plk")    # unbalanced
df_balance = pd.read_pickle("all_train_rebalanced.plk")    # balanced
df_balance  = df_balance.iloc[0:10000]
df_test   = pd.read_pickle("test_and_valid_rebalanced.plk")  # balanced
b_use_separate_test_file = True #False #True #False#True#False #True
#################################

#################################
##### modelfiles
model_in_json = 'model_rtest_e3.json'
model_in_h5   = "model_rtest_e3.h5"
#model_in_h5   = "checkpoints/w_01_0.74.hdf5"
#################################

################################
## Plot Sentence Length
# if b_plot:
#     t_lens = df_total.iloc[:,6].str.len()
#     fig = plt.figure(figsize=(10, 10))
#     ax = plt.subplot(111)
#     ax.set_yscale("log")
#     plt.xlabel('Sentence length')
#     plt.ylabel('Number of sentences')
#     plt.hist(t_lens, bins=40)
#     plt.show()
###############################

###############################
# Pad sequences with to Max Length
#   and Truncate at max length
# pad w/ value 265, since integer tokens range from 0 to 266
#need to reset the index
df_total.index =range(len(df_total))
df_balance.index = range(len(df_balance))
if b_use_separate_test_file:
    df_test.index = range(len(df_test))

### Define padding
def pad_length(this_row,MAXLEN = 10):
    a_padded = pad_sequences([this_row],maxlen=MAXLEN,truncating="post", padding="post",value=265)

    return a_padded[0]
###

l_seq = [f's{i}' for i in range(0,_MAXLEN)]
df_total[l_seq] = pd.DataFrame(pad_sequences(df_total.Tokens,maxlen=_MAXLEN,truncating="post", padding="post",value=265), columns = l_seq)

df_balance[l_seq] = pd.DataFrame(pad_sequences(df_balance.Tokens,maxlen=_MAXLEN,truncating="post", padding="post",value=265), columns = l_seq)
if b_use_separate_test_file:
    df_test[l_seq] = pd.DataFrame(pad_sequences(df_test.Tokens,maxlen=_MAXLEN,truncating="post", padding="post",value=265), columns = l_seq)

### END Padding
###############################

#convert df to arrays
#df_total = df_total.iloc[0:1]
padded_tokens = df_total[l_seq].values
bal_tokens    = df_balance[l_seq].values
l_label_col = ['CWE-119', 'CWE-120', 'CWE-469', 'CWE-476', 'CWE-other']

if b_binary:
    labels = to_categorical(np.asarray(df_total['any-vuln']))
    bal_labels = to_categorical(np.asarray(df_balance['any-vuln']))
else:
    labels = np.asarray(df_total[l_label_col])

if b_use_separate_test_file:
    if b_binary:
        test_labels = to_categorical(np.asarray(df_test['any-vuln']))
    else:
        test_labels = np.asarray(df_test[l_label_col])

# get number of test rows
N_test = int(FRAC_TEST * padded_tokens.shape[0])

# initialze embedding weights to random
np.random.seed(1)
w_embed = np.random.rand(TOKEN_UNIVERSE, TOKEN_UNIVERSE)

if b_use_separate_test_file:
    train_X = padded_tokens
    train_y = labels
    test_y = test_labels
    test_X = df_test[l_seq].values
else:
    train_X = padded_tokens[:-N_test]
    train_y = labels[:-N_test]
    test_X  = padded_tokens[-N_test:]
    test_y  = labels[-N_test:]

# Train new model, or load from file
if not b_load_model_from_file:
    model = NLP_model(b_binary, len(labels[0]), _MAXLEN, TOKEN_UNIVERSE, w_embed)
    model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['acc'])
        
    print(model.summary())
    #    plot_model(model, to_file='schematic.svg',show_layer_names=False)
    #model.fit(train_X, train_y, epochs=N_epoch, batch_size=128, validation_data=(test_X, test_y),class_weight=class_weights)

    if(b_start_from_previous_model):
        model.load_weights(model_in_h5)

    ###add checkpoints
    m_checkpoints   = ModelCheckpoint("checkpoints/w_{epoch:02d}_{val_acc:.2f}.hdf5",
                                      monitor='val_acc', verbose=1, save_best_only=True)
    
    model.fit(train_X, train_y, epochs=N_epoch, batch_size=128, validation_data=(test_X, test_y),callbacks=[m_checkpoints],class_weight=class_weights)
else:
    #load model
    with open(model_in_json,'r') as model_file:
        js_model = model_file.read()
        model = model_from_json(js_model)
        # load weight
    model.load_weights(model_in_h5)
    model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['acc',REC])
    print(model.summary())

l_cm_labels = [True,False]
    
if b_binary:
    predict = model.predict(test_X)
    y_pred   = predict[:,1] > 0.5
    y_1D_test = test_y[:,1] == 1
    
    a_FN = np.logical_and(np.logical_not(y_pred),y_1D_test)
    
    cm = confusion_matrix(y_1D_test,y_pred,labels=l_cm_labels)
    print(cm)
    df_test['FN'] = pd.DataFrame(a_FN)
    df_FN = df_test.loc[df_test.FN==True]
    plt_histogram(df_FN)
    
    precision = cm[0,0] / (cm[0,0]+cm[0,1])
    print(" p: ",precision)
    recall    = cm[0,0] / (cm[0,0]+cm[1,0])
    print(" r: ",recall)
    final_acc       = cm[0,0]+cm[1,1]
    if len(y_1D_test) != 0.:
        final_acc = final_acc/len(y_1D_test)

    print(" a: ",final_acc)

    #plot the CWEs that are causing FNs
    t_FN = df_FN.iloc[:,6].str.len()
    fig = plt.figure(figsize=(10, 10))
    ax = plt.subplot(111)
    ax.set_yscale("log")
    plt.xlabel('Sentence length')
    plt.ylabel('Number of sentences')
    plt.hist(t_FN, bins=40)
    plt.show()
    
    # calulate MCC
    #MCC = matthews_corrcoef(y_1D_test,y_pred)
    TP = cm[0,0]
    TN = cm[1,1]
    FP = cm[1,0]
    FN = cm[0,1]
    
    MCC = ((TP*TN)-(FP*FN))/np.sqrt((TP+FP)*(TP+FN)*(TN+FP)*(TN+FN))
    print(" MCC: ",MCC)

    #calculate F1
    F1  = f1_score(y_1D_test,y_pred,
                   labels=l_cm_labels)
    print(" F1: ",F1)
    
else:
    predict = model.predict(test_X)
    predict_tv =  predict > 0.5

    test_any_vuln = np.logical_or(np.logical_or(np.logical_or(np.logical_or(test_y[:,0],test_y[:,1]),test_y[:,2]),test_y[:,3]) ,test_y[:,4])
    test_no_vuln  = np.logical_not(test_any_vuln)

    pred_any_vuln = np.logical_or(np.logical_or(np.logical_or(np.logical_or(predict_tv[:,0],predict_tv[:,1]),predict_tv[:,2]),predict_tv[:,3]) ,predict_tv[:,4])
    pred_no_vuln  = np.logical_not(pred_any_vuln)

    for ijk in range(0,5):
        y_pred   = predict[:,ijk] > 0.5
        y_1D_test = test_y[:,ijk] == 1
        print("++++++ ",l_label_col[ijk]," +++++")
        cm = confusion_matrix(y_1D_test,y_pred,labels=l_cm_labels)
        cm = cm/len(y_1D_test)
        print(cm)
        precision = cm[0,0] / (cm[0,0]+cm[0,1])
        print(" p: ",precision)
        recall    = cm[0,0] / (cm[0,0]+cm[1,0])
        print(" r: ",recall)

    print("++++++ Any Vulnerability +++++")
    cm = confusion_matrix(test_any_vuln,pred_any_vuln,labels=l_cm_labels)
    cm = cm/len(test_any_vuln)
    print(cm)
    precision = cm[0,0] / (cm[0,0]+cm[0,1])
    print(" p: ",precision)
    recall    = cm[0,0] / (cm[0,0]+cm[1,0])
    print(" r: ",recall)


##############################
### save model to file
if (not b_load_model_from_file) and b_save_model:
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
