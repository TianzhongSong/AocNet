from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Input, GlobalAveragePooling2D
from keras.layers import Cropping2D, concatenate, add, Activation, ZeroPadding2D
from keras.models import Model
from keras import backend as K


def all_one_conv(inputs, filters=8, output_channel=64, padding=0, kernel_size=(3,3), strides=(2,2)):
    stride = kernel_size[0]
    if padding != 0:
        inputs = ZeroPadding2D((padding, padding))(inputs) 
    x = Conv2D(filters, (1, 1), strides=(stride, stride))(inputs)
    feat = []
    feat.append(x)
    for i in range(1, kernel_size[0]*kernel_size[1]):
        crop_x = int(i % kernel_size[0])
        crop_y = int(i // kernel_size[1])
        x = Cropping2D(cropping=((crop_y, 0),(crop_x, 0)))(inputs)
        x = Conv2D(filters, (1, 1), strides=(stride, stride))(x)
        feat.append(x)
    out = concatenate(feat, axis=-1)
    out = Activation("relu")(out)
    out = Conv2D(output_channel, (1, 1), strides=(1, 1))(out)
    return out


def aocNet():
    inputs = Input(shape=(32, 32, 3))

    x = all_one_conv(inputs, filters=16, padding=2, output_channel=128)

    x = all_one_conv(x, filters=32, output_channel=512)

    x = GlobalAveragePooling2D()(x)

    x = Dense(10, activation='softmax')(x)

    model = Model(inputs=inputs, outputs=x)

    return model
