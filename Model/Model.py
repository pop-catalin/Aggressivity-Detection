from keras.applications import VGG16
from keras.models import Model as model
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
import keras

class Model:
    def __init__(self):
        self.__nrFrames = 10
        self.__outputSize = 4096
        self.__CNNmodel = VGG16(weights='imagenet', include_top=True)
        self.__TransferModel = model(inputs=self.__CNNmodel.input,
                             outputs=self.__CNNmodel.get_layer('fc2').output)
        self.__LSTMmodel, optimizer = self.createLSTMmodel()
        self.__LSTMmodel.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    
    def createLSTMmodel(self):
        lstm = Sequential()
        lstm.add(LSTM(1024, input_shape=(self.__nrFrames, self.__outputSize)))
        lstm.add(Dense(1024, activation='relu'))
        lstm.add(Dropout(0.5))
        lstm.add(Dense(1, activation='sigmoid'))
        
        optimizer = keras.optimizers.Adam(lr=0.0001)
        return lstm, optimizer
        
    def getCNNmodel(self):
        return self.__CNNmodel
    
    def getLSTMmodel(self):
        return self.__LSTMmodel
    
    def getTransferModel(self):
        return self.__TransferModel
    
    def getNrFrames(self):
        return self.__nrFrames
    
    def getOutputSize(self):
        return self.__outputSize
    
    def setNrFrames(self, nrFrames):
        self.__nrFrames = nrFrames
        
    def setOutputSize(self, outputSize):
        self.__outputSize = outputSize