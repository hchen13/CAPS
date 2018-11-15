import numpy as np
from keras import Model, Input, optimizers
from keras import backend as K
from keras.layers import Dense, LSTM, Dropout, Conv1D, MaxPooling1D, Flatten, Concatenate, AveragePooling1D, \
    BatchNormalization, regularizers


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

def conv(filters, kernel_size, lambd=0.001):
    return Conv1D(filters, kernel_size, padding='same', activation='relu', kernel_initializer='he_normal', kernel_regularizer=regularizers.l2(lambd))


def direction_inception_model(input_shape, lambd):
    def inception(layer, out_channels):
        num_channels = int(out_channels / 2)

        # equivalence to the 1x1 convolution
        conv1 = conv(num_channels, 1, lambd)(layer)

        conv3 = conv(32, 1, lambd)(layer)
        conv3 = conv(int(num_channels / 2), 3, lambd)(conv3)

        conv6 = conv(16, 1, lambd)(layer)
        conv6 = conv(int(num_channels / 4), 6, lambd)(conv6)

        pool = MaxPooling1D(strides=1, padding='same')(layer)
        pool = conv(int(num_channels / 4), 1, lambd)(pool)

        incep = Concatenate(axis=2)([conv1, conv3, conv6, pool])
        return incep

    inputs = Input(shape=input_shape[1:])

    f = conv(16, 6, lambd)(inputs)
    f = conv(16, 6, lambd)(f)
    p = MaxPooling1D()(f)
    p = BatchNormalization()(p)

    f = conv(32, 1, lambd)(p)
    f = conv(32, 6, lambd)(f)
    p = MaxPooling1D()(f)
    p = BatchNormalization()(p)

    p = inception(p, 128)
    p = BatchNormalization()(p)

    f = conv(128, 1, lambd)(p)
    f = conv(128, 3, lambd)(f)
    p = MaxPooling1D()(f)
    p = BatchNormalization()(p)

    p = inception(p, 256)
    p = BatchNormalization()(p)

    f = conv(256, 1, lambd)(p)
    f = conv(256, 3, lambd)(f)
    p = MaxPooling1D()(f)

    f = conv(512, 1, lambd)(p)
    f = conv(1024, 3, lambd)(f)
    p = MaxPooling1D()(f)
    p = BatchNormalization()(p)

    p = inception(p, 1024)
    p = BatchNormalization()(p)

    feature_vec = Flatten(name='bottleneck')(p)
    dense = Dense(800, activation='relu')(feature_vec)
    dense = Dropout(.2)(dense)

    d = Dense(1, activation='sigmoid', name='direction')(dense)

    model = Model(inputs=inputs, outputs=d)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc', precision, recall, f1_score])
    model.summary()
    return model


def direction_vgg_model(input_shape, keep_prob=.2):
    inputs = Input(shape=input_shape[1:])

    def conv(filters, kernel_size, lambd=0.001):
        return Conv1D(filters, kernel_size, padding='same', activation='relu', kernel_initializer='he_normal', kernel_regularizer=regularizers.l2(lambd))

    f = conv(16, 6)(inputs)
    f = conv(16, 6)(f)
    p = MaxPooling1D()(f)
    p = Dropout(keep_prob)(p)

    f = conv(32, 1)(p)
    f = conv(32, 6)(f)
    f = conv(32, 6)(f)
    p = MaxPooling1D()(f)
    p = Dropout(keep_prob)(p)

    f = conv(64, 1)(p)
    f = conv(64, 3)(f)
    f = conv(64, 3)(f)
    p = MaxPooling1D()(f)
    p = Dropout(keep_prob)(p)

    f = conv(128, 1)(p)
    f = conv(128, 3)(f)
    f = conv(128, 3)(f)
    p = MaxPooling1D()(f)
    p = Dropout(keep_prob)(p)

    f = conv(256, 1)(p)
    f = conv(256, 3)(f)
    f = conv(256, 3)(f)
    p = MaxPooling1D()(f)
    p = Dropout(keep_prob)(p)

    f = conv(512, 1)(p)
    f = conv(512, 3)(f)
    f = Dropout(keep_prob)(f)

    f = conv(512, 3)(f)
    p = AveragePooling1D()(f)
    p = Dropout(keep_prob)(p)

    feature_vec = Flatten(name='bottleneck')(p)

    d = Dense(
        1, activation='sigmoid', name='direction',
        kernel_regularizer=regularizers.l2(.003)
    )(feature_vec)

    model = Model(inputs=inputs, outputs=d)
    from keras.optimizers import SGD
    sgd = SGD()

    def biased_loss(y, y_hat):
        from keras.losses import binary_crossentropy
        return (2 - precision(y, y_hat)) * binary_crossentropy(y, y_hat)

    model.compile(optimizer='adam', loss=biased_loss, metrics=['acc', precision, recall, f1_score])
    model.summary()
    return model


def direction_lstm2(input_shape, lambd=0.0001):
    inputs = Input(shape=input_shape[1:])

    def lstm(units, return_sequences=False):
        return LSTM(
            units, return_sequences=return_sequences,
            kernel_regularizer=regularizers.l2(lambd),
            recurrent_regularizer=regularizers.l2(lambd),
            activity_regularizer=regularizers.l2(lambd)
        )

    f = lstm(32, return_sequences=True)(inputs)
    f = Dropout(.2)(f)
    f = lstm(64)(f)

    # f = Dense(256, kernel_regularizer=regularizers.l2(lambd), activation='relu')(f)
    d = Dense(1, activation='sigmoid', kernel_regularizer=regularizers.l2(lambd))(f)
    model = Model(inputs=inputs, outputs=d)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc', precision, recall])
    model.summary()
    return model
