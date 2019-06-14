from keras.applications.mobilenetv2 import MobileNetV2
from keras.layers import Dense, Input, Dropout
from keras.models import Model
from keras import backend as K
from keras.layers import Dense, GlobalAveragePooling2D
from keras.layers.normalization import BatchNormalization


class MartinMobileNetV2:
    @staticmethod
    def build(width, height, depth, classes):
        input_shape = (width, height, depth)
        chanDim = -1
        if K.image_data_format() == "channels_first":
                inputShape = (depth, height, width)
                chanDim = 1

        base_model = MobileNetV2(
            include_top=False,
            weights='imagenet',
            input_shape=input_shape)

        for layer in base_model.layers:
            layer.trainable = True             

        x = base_model.output
        x = GlobalAveragePooling2D()(x)
        x = Dense(1024, activation='relu')(x)
        x = BatchNormalization()(x)
        x = Dropout(0.2)(x)
        x = Dense(classes, activation='softmax')(x)
        prediction = x

        model = Model(inputs=base_model.input, outputs=prediction)

        return model
