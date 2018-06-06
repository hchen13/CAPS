import numpy as np
from keras import Model, Input
from keras import backend as K
from keras.layers import Dense, LSTM, Dropout, multiply, Conv1D, MaxPooling1D, Flatten


def future_price_lstm(input_shape, lstm_neurons=128, keep_prob=.2):

    inputs = Input(shape=input_shape[1:])

    lstm = LSTM(lstm_neurons, return_sequences=True)(inputs)
    drop = Dropout(keep_prob)(lstm)

    lstm = LSTM(lstm_neurons, return_sequences=False)(drop)
    drop = Dropout(keep_prob)(lstm)

    dense = Dense(128, kernel_initializer='uniform', activation='relu')(drop)

    dense = Dense(32, activation='relu')(dense)
    price = Dense(1, activation='linear')(dense)

    model = Model(inputs=inputs, outputs=price)
    model.compile(optimizer='adam', loss=['mae'], metrics=['mae'])
    model.summary()
    return model

def future_direction_lstm(input_shape, lstm_neurons=128, keep_prob=.2):

    inputs = Input(shape=input_shape[1:])

    lstm = LSTM(lstm_neurons, return_sequences=True)(inputs)
    drop = Dropout(keep_prob)(lstm)

    lstm = LSTM(lstm_neurons, return_sequences=False)(drop)
    drop = Dropout(keep_prob)(lstm)

    dense = Dense(128, kernel_initializer='uniform', activation='relu')(drop)

    # dense = Dense(32, activation='relu')(dense)
    d = Dense(1, activation='sigmoid', name='direction')(dense)

    model = Model(inputs=inputs, outputs=d)
    model.compile(optimizer='adam', loss=['binary_crossentropy'], metrics=['acc'])
    model.summary()
    return model


def future_direction_conv(input_shape):
    inputs = Input(shape=input_shape[1:])

    f = Conv1D(16, 6, padding='valid', activation='relu')(inputs)
    p = MaxPooling1D()(f)

    f = Conv1D(32, 6, padding='valid', activation='relu')(p)
    p = MaxPooling1D()(f)

    f = Conv1D(64, 6, padding='valid', activation='relu')(p)
    p = MaxPooling1D()(f)

    f = Conv1D(128, 6, padding='same', activation='relu')(p)
    p = MaxPooling1D()(f)

    f = Conv1D(256, 6, padding='same', activation='relu')(p)
    p = MaxPooling1D()(f)

    feature_vec = Flatten()(p)

    d = Dense(1, activation='sigmoid', name='direction')(feature_vec)

    model = Model(inputs=inputs, outputs=d)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])
    model.summary()
    return model

