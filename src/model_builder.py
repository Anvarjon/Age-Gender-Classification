from keras import layers, regularizers
from tensorflow import keras
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

class ModelBuilder:
    def __init__(self, inp_shape, num_classes, reg_value):
        self.inp_shape = inp_shape
        self.num_classes = num_classes
        self.setRegularizer(reg_value)
    
    def setRegularizer(self, reg_value):
        if reg_value != None:
            self.reg = regularizers.l2(l2=reg_value)
        else:
            self.reg = None
    
    def FLB(self, inp):
        # Feature learning block (FLB)
        x=layers.Conv2D(filters = 120, kernel_size = (9, 9), strides=(2, 2), activation='relu', kernel_regularizer=self.reg, padding='same')(inp)
        x=layers.BatchNormalization()(x)
        x=layers.MaxPooling2D(pool_size=(2, 2),strides=(2, 2))(x)
        x=layers.Conv2D(filters = 256, kernel_size = (5, 5), strides=(1, 1), activation='relu', kernel_regularizer=self.reg, padding='same')(x)
        x=layers.MaxPooling2D(pool_size=(2, 2))(x)
        x=layers.Conv2D(filters = 384, kernel_size = (3, 3), activation='relu', kernel_regularizer=self.reg, padding='same')(x)
        x=layers.BatchNormalization()(x)
        return x

    def time_attention(self, inp):
        x=layers.Conv2D(filters = 64, kernel_size = (1, 9), activation='relu', kernel_regularizer=self.reg, padding='same')(inp)
        x=layers.Conv2D(filters = 64, kernel_size = (1, 3), activation='relu', kernel_regularizer=self.reg, padding='same')(x)
        x=layers.Conv2D(filters = 64, kernel_size = (1, 3), activation='relu', kernel_regularizer=self.reg, padding='same')(x)
        x=layers.BatchNormalization()(x)
        return x

    def frequency_attention(self, inp):
        x=layers.Conv2D(filters = 64, kernel_size = (9, 1), activation='relu', kernel_regularizer=self.reg, padding='same')(inp)
        x=layers.Conv2D(filters = 64, kernel_size = (3, 1), activation='relu', kernel_regularizer=self.reg, padding='same')(x)
        x=layers.Conv2D(filters = 64, kernel_size = (3, 1), activation='relu', kernel_regularizer=self.reg, padding='same')(x)
        x=layers.BatchNormalization()(x)
        return x

    def MAM(self, inp):
        # Time attention module
        ta = self.time_attention(inp)
        # Frequency attention module
        fa = self.frequency_attention(inp)
        # Concatenate time and frequency attention module outputs to build MAM
        mam=layers.concatenate([ta, fa])
        mam=layers.BatchNormalization()(mam)
        return mam

    def build_model(self, show_summary=True):
        inp = keras.Input(shape=self.inp_shape)
        # First feature learning block (FLB-1)
        x = self.FLB(inp)
        # Multi-attention module (MAM)
        mam = self.MAM(x)
        # Concatenate FLB-1 and MAM outputs
        x = layers.concatenate([x, mam])
        # Second feature learning block (FLB-2)
        x = self.FLB(x)
        x = layers.Flatten()(x)
        x = layers.Dense(80, activation='relu', kernel_regularizer=self.reg)(x)
        x = layers.BatchNormalization()(x)
        out = layers.Dense(units=self.num_classes, activation='softmax')(x)
        model = keras.Model(inp, out)
        if show_summary:
            print(model.summary(line_length=120))
        return model