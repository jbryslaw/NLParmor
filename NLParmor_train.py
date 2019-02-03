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
#import plot_confusion_matrix
from NLParmor_model import NLP_model
import itertools
import losses

#################################
### Consts
_MAXLEN = 200 #max function length in number of tokens
#_MAXLEN = 100 #max function length in number of tokens
#_MAXLEN = 400 #max function length in number of tokens 
_MAXLEN = _MAXLEN
FRAC_TEST=.2 #Fraction of data to set aside for testing
TOKEN_UNIVERSE = 266 # Number of unique tokens
N_epoch = 100
#N_epoch = 2
# there are 29 functions without vulnerabilites
#    to every 2 vulnerable functions
#class_weights = {0:29,1:2}
#class_weights = {0:2,1:29}
#class_weights = {0:2,1:50}
class_weights = {0:1,1:1}
#################################

#################################
### switches
b_floss     = False#True #False #True #True #False
b_plot      = True #False #True #False
b_load_model_from_file = True #False #True#False#True#False #True #False #True #False #True #False
# do a binary classification (yes/no vuln) or categorical classification
b_binary    = True #False
b_start_from_previous_model = True
#################################

#################################
### training file
# df_total  = pd.read_pickle("train0_rebalenced.plk")
#df_total  = pd.read_pickle("all_rebalenced.plk")
#df_total  = pd.read_pickle("all_but_test_rebalenced.plk")
#df_total  = pd.read_pickle("all_but_test.plk")    # unbalanced
#df_total  = pd.read_pickle("all_train.plk")    # unbalanced
df_total  = pd.read_pickle("all_train_rebalanced.plk")    # balanced
#df_total  = df_total.iloc[0:10000]
df_balance = pd.read_pickle("all_train_rebalanced.plk")    # balanced
df_balance  = df_balance.iloc[0:10000]
#df_total  = df_total.iloc[0:10000]
#df_test   = pd.read_pickle("test_rebalenced.plk") # balanced
df_test   = pd.read_pickle("test_and_valid_rebalanced.plk")  # balanced
# df_test   = pd.read_pickle("all_train.plk")  # balanced
# df_test    = df_test.iloc[0:100000]
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

# Focal Loss (1708.02002)
def floss():
    def Floss(truth,reco):
        # g -> floss power, a --> class weighting
        # g=0, a=1 should produce normal CE
        g=1.25
        a=0.25
        #a=0.25
        clip = 1e15 #1e-9
        reco = kb.clip(reco,clip,1-clip)
        pT0 = tf.where(tf.equal(truth, 0), reco, tf.ones_like(reco))
        pT1 = tf.where(tf.equal(truth, 1), reco, tf.ones_like(reco))

        loss = (-1)*kb.sum(a*kb.pow(1-pT1,g)*kb.log(pT1))-(kb.sum((1-a)*kb.pow(pT0,g)*kb.log(1-pT0)) )
        # pT1 =  kb.clip(pT1,clip,1-clip)
        # pT0 =  kb.clip(pT1,clip,1-clip)
        # loss = (-1)*kb.mean(a*kb.pow(1-pT1,g)*kb.log(pT1))-(kb.mean((1-a)*kb.pow(pT0,g)*kb.log(1-pT0)) )

        return loss
    return Floss

# Train new model, or load from file
if not b_load_model_from_file:
    model = NLP_model(b_binary, len(labels[0]), _MAXLEN, TOKEN_UNIVERSE, w_embed)
    if b_floss:
        #train with a few epochs of CE before trying floss
        model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['acc'])
        #model.fit(train_X, train_y, epochs=3, batch_size=128, validation_data=(test_X, test_y),class_weight=class_weights)
        model.fit(bal_tokens, bal_labels, epochs=1, batch_size=128, validation_data=(test_X, test_y),class_weight=class_weights)

        model.compile(loss=floss(),optimizer='SGD',metrics=['acc'])
        #model.compile(loss=losses.focal(),optimizer='SGD',metrics=['acc'])
    else:
        model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['acc'])
        
    print(model.summary())
    #    plot_model(model, to_file='schematic.svg',show_layer_names=False)
    #model.fit(train_X, train_y, epochs=N_epoch, batch_size=128, validation_data=(test_X, test_y),class_weight=class_weights)

    if(b_start_from_previous_model):
        model.load_weights(model_in_h5)

    ### ###add checkpoints
    m_checkpoints   = ModelCheckpoint("checkpoints/w_{epoch:02d}_{val_acc:.2f}.hdf5",
                                      monitor='val_acc', verbose=1, save_best_only=True)

    
    model.fit(train_X, train_y, epochs=N_epoch, batch_size=128, validation_data=(test_X, test_y),callbacks=[m_checkpoints])
else:
    #load model
    with open(model_in_json,'r') as model_file:
        js_model = model_file.read()
        model = model_from_json(js_model)
        # load weight
    model.load_weights(model_in_h5)
    model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['acc'])
    print(model.summary())

l_cm_labels = [True,False]
#l_cm_labels = [True,False]
#l_cm_labels = [1,0]
    
if b_binary:
    predict = model.predict(test_X)
    y_pred   = predict[:,1] > 0.5
    y_1D_test = test_y[:,1] == 1 
    cm = confusion_matrix(y_1D_test,y_pred,labels=l_cm_labels)
    # if len(y_1D_test) != 0.:
    #     cm = cm/len(y_1D_test)
    print(cm)

    precision = cm[0,0] / (cm[0,0]+cm[0,1])
    print(" p: ",precision)
    recall    = cm[0,0] / (cm[0,0]+cm[1,0])
    print(" r: ",recall)
    final_acc       = cm[0,0]+cm[1,1]
    if len(y_1D_test) != 0.:
        final_acc = final_acc/len(y_1D_test)

    print(" a: ",final_acc)
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



def plot_cm(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()


## ###plot cm
if b_plot:
    np.set_printoptions(precision=2)

    plt.figure()
    class_names = ['Vulnerable','Not Vulnerable']
    plot_cm(cm, classes=class_names, normalize=True,
                           title='Normalized confusion matrix')
    plt.show()


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
