import keras
from keras.layers import Conv1D, MaxPooling1D, Embedding, BatchNormalization, Dense, Input, Flatten, Dropout
from keras.models import Model

def NLP_model(b_binary, labels_index, max_length, TOKEN_UNIVERSE, Embed ):
    st_act = 'relu'
    
    # Yoon Kim model 1408.5882
    ly_in = Input(shape=(max_length,), dtype='int32')
    ly_embed = Embedding(TOKEN_UNIVERSE,
                           TOKEN_UNIVERSE,
                           input_length=max_length,
                           weights=[Embed],                           
                           trainable=True)(ly_in)        

    if b_binary:
        filters = [7,12,15,17]
        poolsize = [7,12,15,17]
        N_filters = 256 #1024
    else:
        filters = [7,12,15]
        poolsize = [7,12,15]
        N_filters = 256 #1024

    ly_N_conv = []
    
    for i_filter in range(len(filters)):
        i_ly_conv = Conv1D(filters=N_filters, kernel_size=filters[i_filter], activation=st_act)(ly_embed)
        i_ly_batchN = BatchNormalization()(i_ly_conv)
        i_ly_pool = MaxPooling1D(pool_size=poolsize[i_filter])(i_ly_batchN)
        ly_N_conv.append(i_ly_pool)
        
    ly_merge = keras.layers.Concatenate(axis=1)(ly_N_conv)

    # ly_final_conv = Conv1D(filters=128, kernel_size=3, activation=st_act)(ly_merge)
    # ly_final_pool = MaxPooling1D(pool_size=3)(ly_final_pool)
    # ly_final_drop = Dropout(0.5)(ly_final_drop)
    #    ly_final_batchN = BatchNormalization()(ly_merge)
    ly_drop = Dropout(0.5)(ly_merge)
    ly_flat = Flatten()(ly_drop)

    ly_dense = Dense(128, activation=st_act)(ly_flat)

    if b_binary:
        ly_out = Dense(labels_index, activation='softmax')(ly_dense)
    else:
        ly_out = Dense(labels_index, activation='sigmoid')(ly_dense)
    model = Model(ly_in, ly_out)
    
    return model
