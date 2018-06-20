import numpy as np
from keras import Model, Input
from keras import backend as K
from keras.layers import Dense, LSTM, Dropout, Conv1D, MaxPooling1D, Flatten, Concatenate, AveragePooling1D


def elliptic_paraboloid_weight(x, y, diff_weight, same_weight):
    """This function produces a coefficient between x and y based on rotated elliptic
    paraboloid function, which suggests a large value if x and y are in different
    directions and a small value otherwise.

    :param x: first value, could be a numpy array
    :param y: seconde value, should have the same shape as `x`
    :param diff_weight: the penalty weight for different direction
    :param same_weight: the penalty weight for same direction
    :return: a coefficient
    """
    t = -np.pi / 4  # rotate angle

    x_rot = x * np.cos(t) + y * np.sin(t)
    y_rot = -x * np.sin(t) + y * np.cos(t)

    z = x_rot ** 2 / diff_weight + y_rot ** 2 / same_weight

    return z


def directional_loss(y, y_hat):
    squared_error = K.mean(K.square(y - y_hat))
    diff_sign = y * y_hat * 4
    sign_error = 2 - K.sigmoid(diff_sign)
    return squared_error * sign_error


def directional_accuracy(y, y_hat):
    return K.mean(y * y_hat > 0)


def precision(y, y_hat):
    true_positives = K.sum(K.round(K.clip(y * y_hat, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_hat, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision


def recall(y, y_hat):
    true_positives = K.sum(K.round(K.clip(y * y_hat, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y, 0, 1)))
    return true_positives / (possible_positives + K.epsilon())


def f1_score(y, y_hat):
    p = precision(y, y_hat)
    r = recall(y, y_hat)
    return 2 / (1 / p + 1 / r)



def future_price_conv(input_shape, keep_prob=.2):
    inputs = Input(shape=input_shape[1:])

    f = Conv1D(16, 6, padding='valid', activation='relu')(inputs)
    f = Conv1D(16, 6, padding='same', activation='relu')(f)
    p = MaxPooling1D()(f)
    p = Dropout(keep_prob)(p)

    f = Conv1D(32, 6, padding='valid', activation='relu')(p)
    f = Conv1D(32, 6, padding='same', activation='relu')(f)
    p = MaxPooling1D()(f)
    p = Dropout(keep_prob)(p)

    f = Conv1D(64, 6, padding='valid', activation='relu')(p)
    f = Conv1D(64, 6, padding='same', activation='relu')(f)
    p = MaxPooling1D()(f)
    p = Dropout(keep_prob)(p)

    f = Conv1D(128, 6, padding='same', activation='relu')(p)
    f = Conv1D(128, 6, padding='same', activation='relu')(f)
    p = MaxPooling1D()(f)
    p = Dropout(keep_prob)(p)

    f = Conv1D(256, 6, padding='same', activation='relu')(p)
    f = Conv1D(256, 6, padding='same', activation='relu')(f)
    p = MaxPooling1D()(f)
    p = Dropout(keep_prob)(p)

    feature_vec = Flatten()(p)

    p = Dense(1, activation='linear', name='price')(feature_vec)

    model = Model(inputs=inputs, outputs=p)
    model.compile(optimizer='adam', loss=directional_loss, metrics=[directional_accuracy])
    model.summary()
    return model


def continuous_price_model(pretrained, mode='transfer'):
    if mode == 'transfer':
        for layer in pretrained.layers:
            layer.trainable = False
    x = pretrained.get_layer(name='bottleneck').output
    hidden = Dense(64, activation='relu', kernel_initializer='uniform')(x)
    y = Dense(1, activation='linear', name='price')(hidden)

    model = Model(inputs=pretrained.inputs, outputs=y)
    model.compile(optimizer='adam', loss=directional_loss, metrics=[directional_accuracy])
    model.summary()
    return model


def direction_inception_model(input_shape, keep_prob=.2):

    def inception(layer, out_channels):
        # equivalence to the 1x1 convolution
        num_channels = int(out_channels / 2)

        conv1 = Conv1D(num_channels, 1, padding='same', activation='relu')(layer)

        conv3 = Conv1D(int(num_channels / 4), 1, padding='same', activation='relu')(layer)
        conv3 = Conv1D(int(num_channels / 2), 3, padding='same', activation='relu')(conv3)

        conv6 = Conv1D(int(num_channels / 8), 1, padding='same', activation='relu')(layer)
        conv6 = Conv1D(int(num_channels / 4), 6, padding='same', activation='relu')(conv6)

        pool = MaxPooling1D(strides=1, padding='same')(layer)
        pool = Conv1D(int(num_channels / 4), 1, padding='same', activation='relu')(pool)

        incep = Concatenate(axis=2)([conv1, conv3, conv6, pool])
        return incep


    inputs = Input(shape=input_shape[1:])

    f = Conv1D(16, 6, padding='same', activation='relu')(inputs)
    f = Conv1D(32, 3, padding='same', activation='relu')(f)
    f = Conv1D(64, 3, padding='same', activation='relu')(f)
    p = MaxPooling1D()(f)
    p = Dropout(keep_prob)(p)

    # inception layers
    incep = inception(p, 128)
    incep = inception(incep, 256)
    p = MaxPooling1D()(incep)
    p = Dropout(keep_prob)(p)

    incep = inception(p, 480)
    incep = inception(incep, 512)
    p = MaxPooling1D()(incep)
    p = Dropout(keep_prob)(p)

    incep = inception(p, 512)
    incep = inception(incep, 800)
    p = MaxPooling1D()(incep)
    p = Dropout(keep_prob)(p)

    # incep = inception(p, 1024)
    # incep = inception(incep, 1024)
    # p = AveragePooling1D()(incep)
    # p = Dropout(keep_prob)(p)

    feature_vec = Flatten(name='bottleneck')(p)
    dense = Dense(800, activation='relu')(feature_vec)

    d = Dense(1, activation='sigmoid', name='direction')(dense)

    model = Model(inputs=inputs, outputs=d)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc', precision, recall, f1_score])
    model.summary()
    return model


def future_direction_conv(input_shape, keep_prob=.2):
    inputs = Input(shape=input_shape[1:])

    f = Conv1D(16, 6, padding='valid', activation='relu')(inputs)
    f = Conv1D(16, 6, padding='same', activation='relu')(f)
    p = MaxPooling1D()(f)
    p = Dropout(keep_prob)(p)

    f = Conv1D(32, 6, padding='valid', activation='relu')(p)
    f = Conv1D(32, 6, padding='same', activation='relu')(f)
    p = MaxPooling1D()(f)
    p = Dropout(keep_prob)(p)

    f = Conv1D(64, 6, padding='valid', activation='relu')(p)
    f = Conv1D(64, 6, padding='same', activation='relu')(f)
    p = MaxPooling1D()(f)
    p = Dropout(keep_prob)(p)

    f = Conv1D(128, 6, padding='same', activation='relu')(p)
    f = Conv1D(128, 6, padding='same', activation='relu')(f)
    p = MaxPooling1D()(f)
    p = Dropout(keep_prob)(p)

    f = Conv1D(256, 6, padding='same', activation='relu')(p)
    f = Conv1D(256, 6, padding='same', activation='relu')(f)
    p = MaxPooling1D()(f)
    p = Dropout(keep_prob)(p)

    feature_vec = Flatten(name='bottleneck')(p)

    d = Dense(1, activation='sigmoid', name='direction')(feature_vec)

    model = Model(inputs=inputs, outputs=d)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc', precision, recall, f1_score])
    model.summary()
    return model


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
