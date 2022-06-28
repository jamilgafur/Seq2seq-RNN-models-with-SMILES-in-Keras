import pandas as pd
import os
import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import Draw, Descriptors
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from keras.models import Model
from keras.layers import Input
from keras.layers import LSTM
from keras.layers import Dense
from keras.layers import Concatenate
from keras import regularizers
from keras.callbacks import History, ReduceLROnPlateau
from keras.optimizers import RMSprop, Adam


def get_test_train(filename="zinc.csv"):
    # csv found here: https://raw.githubusercontent.com/aspuru-guzik-group/chemical_vae/master/models/zinc_properties/250k_rndm_zinc_drugs_clean_3.csv

    data = pd.read_csv(filename)
    train, test =  train_test_split(data["smiles"], random_state=42)
    return data, train, test

def vec_to_string(vector, int_to_char):
    return "".join([int_to_char[idx] for idx in np.argmax(vector, axis=1)])


def vectorize(smiles, embed, charset, char_to_int):
        one_hot =  np.zeros((smiles.shape[0], embed , len(charset)),dtype=np.int8)
        for i,smile in enumerate(smiles):
            #encode the startchar
            one_hot[i,0,char_to_int["!"]] = 1
            #encode the rest of the chars
            for j,c in enumerate(smile):
                one_hot[i,j+1,char_to_int[c]] = 1
            #Encode endchar
            one_hot[i,len(smile)+1:,char_to_int["E"]] = 1
        #Return two, one for input and the other for output
        return one_hot[:,0:-1,:], one_hot[:,1:,:]

def build_model(input_shape, output_dim, latent_dim, lstm_dim):
    unroll = False
    encoder_inputs = Input(shape=input_shape)
    encoder = LSTM(lstm_dim, return_state=True,
                    unroll=unroll)
    encoder_outputs, state_h, state_c = encoder(encoder_inputs)
    states = Concatenate(axis=-1)([state_h, state_c])
    neck = Dense(latent_dim, activation="relu")
    neck_outputs = neck(states)
    decode_h = Dense(lstm_dim, activation="relu")
    decode_c = Dense(lstm_dim, activation="relu")
    state_h_decoded =  decode_h(neck_outputs)
    state_c_decoded =  decode_c(neck_outputs)
    encoder_states = [state_h_decoded, state_c_decoded]
    decoder_inputs = Input(shape=input_shape)
    decoder_lstm = LSTM(lstm_dim,
                        return_sequences=True,
                        unroll=unroll
                    )
    decoder_outputs = decoder_lstm(decoder_inputs, initial_state=encoder_states)
    decoder_dense = Dense(output_dim, activation='softmax')
    decoder_outputs = decoder_dense(decoder_outputs)
    #Define the model, that inputs the training vector for two places, and predicts one character ahead of the input
    return Model([encoder_inputs, decoder_inputs], decoder_outputs),encoder_inputs,  neck_outputs, decode_h, decode_c


def main():
    data, smiles_train, smiles_test = get_test_train()
    # The SMILES must be vectorized to one-hot encoded arrays. To do this a character set is built from all characters found in the SMILES string (both train and test). Also, some start and stop characters are added, which will be used to initiate the decoder and to signal when SMILES generation has stopped. The stop character also work as padding to get the same length of all vectors, so that the network can be trained in batch mode.
    charset = set("".join(list(data.smiles))+"!E")
    # The character set is used to define two dictionaries to translate back and forth between index and character. The maximum length of the SMILES strings is needed as the RNN’s will be trained in batch mode, and is set to the maximum encountered + some extra.
    char_to_int = dict((c,i) for i,c in enumerate(charset))
    int_to_char = dict((i,c) for i,c in enumerate(charset))
    embed = max([len(smile) for smile in data.smiles]) + 5
    
    print("The Charset is: {}".format(str(charset)))
    print("The embedding length is: {}".format(embed))

    # get train data, 
    X_train, Y_train = vectorize(smiles_train.values,  embed, charset, char_to_int)

    # Dat is the string with ! for begining and E for the endding (E for padding)
    print("Vector data is: \n{}".format(X_train[0]))
    print("\n")
    print("Converted to string is: \n\t{}".format(vec_to_string(X_train[0],int_to_char)))
    print("=========")

    # Label is the actual string
    print("Vector label is: \n{}".format(Y_train[0]))
    print("Converted to string is: \n\t{}".format(vec_to_string(Y_train[0],int_to_char)))

    X_test, Y_test = vectorize(smiles_test.values,  embed, charset, char_to_int)
    
    input_shape = X_train.shape[1:]
    output_dim = Y_train.shape[-1]
    latent_dim = 64
    lstm_dim = 64
    model, encoder_inputs, neck_outputs, decode_h, decode_c = build_model(input_shape, output_dim, latent_dim, lstm_dim)
    print("==="*10)
    print(model.summary())
    print("==="*10)

    h = History()
    rlr = ReduceLROnPlateau(monitor='val_loss', factor=.5, patience=10, min_lr=1e-5)
    opt=Adam(lr=0.005) #Default 0.001
    model.compile(optimizer=opt, loss='categorical_crossentropy')
    model.fit([X_train,X_train],Y_train,
                    epochs=20,
                    batch_size=256,
                    shuffle=True,
                    callbacks=[h, rlr],
                    validation_data=[[X_test,X_test],Y_test ])

    plt.plot(h.history["loss"], label="Loss")
    plt.plot(h.history["val_loss"], label="Val_Loss")
    plt.yscale("log")
    plt.legend()
    plt.savefig("loss_val_Loss.png")
    # Get the Encoder
    smiles_to_latent_model = Model(encoder_inputs, neck_outputs)

    # Get the Decoder
    latent_input = Input(shape=(latent_dim,))
    #reuse_layers
    state_h_decoded_2 =  decode_h(latent_input)
    state_c_decoded_2 =  decode_c(latent_input)
    latent_to_states_model = Model(latent_input, [state_h_decoded_2, state_c_decoded_2])
    latent_to_states_model.save("Blog_simple_lat2state.h5")
    #Last one is special, we need to change it to stateful, and change the input shape
    inf_decoder_inputs = Input(batch_shape=(1, 1, input_shape[1]))
    inf_decoder_lstm = LSTM(lstm_dim,
                        return_sequences=True,
                        unroll=False,
                        stateful=True
                    )
    inf_decoder_outputs = inf_decoder_lstm(inf_decoder_inputs)
    inf_decoder_dense = Dense(output_dim, activation='softmax')
    inf_decoder_outputs = inf_decoder_dense(inf_decoder_outputs)
    sample_model = Model(inf_decoder_inputs, inf_decoder_outputs)

    #Transfer Weights
    for i in range(1,3):
        sample_model.layers[i].set_weights(model.layers[i+6].get_weights())
    sample_model.save("Blog_simple_samplemodel.h5")

    # Encode all the test data       
    x_latent = smiles_to_latent_model.predict(X_test)

    molno = 5
    latent_mol = smiles_to_latent_model.predict(X_test[molno:molno+1])
    sorti = np.argsort(np.sum(np.abs(x_latent - latent_mol), axis=1))
    print(sorti[0:10])
    print(smiles_test.iloc[sorti[0:8]])
    Draw.MolsToImage(smiles_test.iloc[sorti[0:8]].apply(Chem.MolFromSmiles))

main()