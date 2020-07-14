import cv2
import os
from random import shuffle
import numpy as np
import h5py
from matplotlib import pyplot as plt

class Controller:
    def __init__(self, m, path):
        self.__model = m
        self.__path = path
        self.__imageSize = 224
        self.__trainingVideos = []
        self.__validationVideos = []
        
    def getTrainingVideos(self):
        return self.__trainingVideos
    
    def getValidationVideos(self):
        return self.__validationVideos
    
    def getModel(self):
        return self.__model
    
    def getPath(self):
        return self.__path
    
    def setPath(self, path):
        self.__path = path
        
    def getImageSize(self):
        return self.__imageSize
    
    def getFrames(self, file):
        print(file)
        frames = []
        cap = cv2.VideoCapture(self.__path + '\\' + file)
        nrOfFrames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        count = 0
        
        while len(frames) < self.__model.getNrFrames():
            ret, frame = cap.read()
            if count % int(nrOfFrames/self.__model.getNrFrames()) == 0:
                if ret == False:
                    break
                else:
                    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    resized = cv2.resize(image, (self.__imageSize, self.__imageSize))
                    frames.append(resized)
            count += 1
        transformedFrames = (np.array(frames) / 255.).astype(np.float16)
        return transformedFrames
    
    def getFramesLongVideo(self, file):
        frames = []
        transformedFrames = []
        cap = cv2.VideoCapture(self.__path + '\\' + file)
        
        nrOfFrames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        nrFps = cap.get(cv2.CAP_PROP_FPS)
        videoLength = int(nrOfFrames / nrFps)       
        
        for i in range(0, int(videoLength/2)):
            count = 0
            
            frames.append([])
            transformedFrames.append([])

            while len(frames[i]) < self.__model.getNrFrames():
                ret, frame = cap.read()
                
                if count % int(videoLength/2) == 0:    
                    
                    if ret == False:
                        break
                    else:
                        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        resized = cv2.resize(image, (self.__imageSize, self.__imageSize))
                        frames[i].append(resized)
                        
                count += 1
                
            transformedFrames[i] = (np.array(frames[i]) / 255.).astype(np.float16)        
        return transformedFrames      
    
    def labelVideos(self):
        namesAndLabels = []
        
        for _, _, names in os.walk(self.__path):
            for name in names:
                if name[0] == 'V' or name[0] == 'f':
                    namesAndLabels.append((name, 1))
                else:
                    namesAndLabels.append((name, 0))
                    
        shuffle(namesAndLabels)          
        return namesAndLabels
    
    def yieldCNNpredictions(self, namesAndLabels):
        for i in namesAndLabels:
            imagesFromOneVideo = self.getFrames(i[0])
            transferOutput = self.__model.getTransferModel().predict(imagesFromOneVideo)
            labels = i[1] * np.ones([10, 1])
            yield transferOutput, labels
    
    def createCNNtrainingFile(self, namesAndLabels):
        predictionsGenerator = self.yieldCNNpredictions(namesAndLabels)
        predictionsSegment = next(predictionsGenerator)
        
        nrOfRows = predictionsSegment[0].shape[0]
        #with h5py.File('trainingfileTEST1label.h5', 'w') as f:
        #with h5py.File('trainingfilevideostogether.h5', 'w') as f:
        #with h5py.File('trainingfiledifferentframes.h5', 'w') as f: 
        with h5py.File('trainingfile.h5', 'w') as file: 
            datasetFeature = file.create_dataset('feature', shape=predictionsSegment[0].shape,
                                       maxshape=(None, 4096), chunks=predictionsSegment[0].shape,
                                       dtype=predictionsSegment[0].dtype)
            datasetLabel = file.create_dataset('label', shape=predictionsSegment[1].shape,
                                       maxshape=(None, 1), chunks=predictionsSegment[1].shape,
                                       dtype=predictionsSegment[1].dtype)
            datasetFeature[:] = predictionsSegment[0]
            datasetLabel[:] = predictionsSegment[1]
            
            for predictionsSegment in predictionsGenerator:
                print('x')
                datasetFeature.resize(nrOfRows + predictionsSegment[0].shape[0], axis=0)
                datasetLabel.resize(nrOfRows + predictionsSegment[1].shape[0], axis=0)

                datasetFeature[nrOfRows:] = predictionsSegment[0]
                datasetLabel[nrOfRows:] = predictionsSegment[1]
                
                nrOfRows += predictionsSegment[0].shape[0]
                
    def createCNNvalidationFile(self, namesAndLabels):
        predictionsGenerator = self.yieldCNNpredictions(namesAndLabels)
        predictionsSegment = next(predictionsGenerator)
        nrOfRows = predictionsSegment[0].shape[0]
        
        #with h5py.File('testfileTEST1label.h5', 'w') as f:
        #with h5py.File('testfilevideostogether.h5', 'w') as f:
        #with h5py.File('testfiledifferentframes.h5', 'w') as f:
        #with h5py.File('testfile10frames2.h5', 'w') as file:
        with h5py.File('validationfile.h5', 'w') as file:
            datasetFeature = file.create_dataset('feature', shape=predictionsSegment[0].shape,
                                       maxshape=(None, 4096), chunks=predictionsSegment[0].shape,
                                       dtype=predictionsSegment[0].dtype)
            datasetLabel = file.create_dataset('label', shape=predictionsSegment[1].shape,
                                       maxshape=(None, 1), chunks=predictionsSegment[1].shape,
                                       dtype=predictionsSegment[1].dtype)
            datasetFeature[:] = predictionsSegment[0]
            datasetLabel[:] = predictionsSegment[1]
            
            for predictionsSegment in predictionsGenerator:
                print('y')
                datasetFeature.resize(nrOfRows + predictionsSegment[0].shape[0], axis=0)
                datasetLabel.resize(nrOfRows + predictionsSegment[1].shape[0], axis=0)

                datasetFeature[nrOfRows:] = predictionsSegment[0]
                datasetLabel[nrOfRows:] = predictionsSegment[1]
                
                nrOfRows += predictionsSegment[0].shape[0]
                
    def getTrainingDataFromFile(self):
        #with h5py.File('trainingfiledifferentframes.h5', 'r') as f:
        with h5py.File('trainingfile.h5', 'r') as file:
            featureBatch = file['feature'][:]
            labelBatch = file['label'][:]
        
        data = []
        targetLabel = []
        count = 0
        
        for i in range(0, int(len(featureBatch)/self.__model.getNrFrames())):
            newCount = count + self.__model.getNrFrames()
            data.append(featureBatch[count:newCount])
            targetLabel.append(np.array(labelBatch[count]))
            count = newCount
            
        return data, targetLabel
        
    def getValidationDataFromFile(self):
        with h5py.File('validationfile.h5', 'r') as file:
        #with h5py.File('testfiledifferentframes.h5', 'r') as f:
            featureBatch = file['feature'][:]
            labelBatch = file['label'][:]
        
        data = []
        targetLabel = []
        count = 0
        
        for i in range(0, int(len(featureBatch)/self.__model.getNrFrames())):
            newCount = count + self.__model.getNrFrames()
            data.append(featureBatch[count:newCount])
            targetLabel.append(np.array(labelBatch[count]))
            count = newCount
            
        return data, targetLabel
    
    def createSets(self):
        labeledVideos = self.labelVideos()  
        trainingSetLength = int(0.8 * len(labeledVideos))
        
        self.__trainingVideos = labeledVideos[:trainingSetLength]
        self.__validationVideos = labeledVideos[trainingSetLength:]
        
    def createCNNfiles(self):
        self.createCNNtrainingFile(self.__trainingVideos)
        self.createCNNvalidationFile(self.__validationVideos)
    
    def retrainModel(self):
        self.createSets()
        self.createCNNfiles()
    
        trainingData, trainingTarget = self.getTrainingDataFromFile()
        validationData, validationTarget = self.getValidationDataFromFile()
        
        history = self.__model.getLSTMmodel().fit(np.array(trainingData), np.array(trainingTarget), epochs=10, batch_size=32, verbose=1,
                                                  validation_data=(np.array(validationData),np.array(validationTarget)))
        
        self.__model.getLSTMmodel().save_weights("modell.h5")
        
    def trainModel(self):
        self.createSets()
        self.createCNNfiles()
        
        trainingData, trainingTarget = self.getTrainingDataFromFile()
        validationData, validationTarget = self.getValidationDataFromFile()
        
        history = self.__model.getLSTMmodel().fit(np.array(trainingData), np.array(trainingTarget), 
                                                  epochs=50, batch_size=32, verbose=1,
                                                  validation_data=(np.array(validationData),np.array(validationTarget)))
        
        plt.plot(history.history['accuracy'])
        plt.plot(history.history['val_accuracy'])
        plt.title('Model Accuracy')
        plt.ylabel('Accuracy')
        plt.xlabel('Epoch')
        plt.legend(['train', 'validation'], loc='upper left')
        plt.show()
        
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('Model Loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(['train', 'validation'], loc='upper left')
        plt.show()
        
        self.__model.getLSTMmodel().save_weights("model.h5")       
    
    def testModel(self):
        self.__model.getLSTMmodel().load_weights("model.h5")
        validationData, validationTarget = self.getValidationDataFromFile()
        
        result = self.__model.getLSTMmodel().evaluate(np.array(validationData), np.array(validationTarget))
        
        for name, value in zip(self.__model.getLSTMmodel().metrics_names, result):
            print(name, value)
    
    def transformFramesWebcam(self, frames):
        resized=[]
        count = 0
        
        for frame in frames:
            if count % 10 == 0:
                resized.append(cv2.resize(frame, (self.__imageSize, self.__imageSize)))
            count += 1
            
        framesArray = (np.array(resized) / 255.).astype(np.float16)
        return framesArray
    
    def transformFramesVideo(self, frames):
        nrOfFrames = len(frames)
        
        resizedFrames = []
        transformedFrames = []        
        
        if nrOfFrames < 100:
            multiply = 2
        else:
            multiply = 10
            
        for i in range(0, int(nrOfFrames / (multiply * self.__model.getNrFrames()))):            
            count = 0
            
            resizedFrames.append([])
            transformedFrames.append([])

            while len(resizedFrames[i]) < self.__model.getNrFrames():          
                if count % multiply == 0:                        
                    resized = cv2.resize(frames[count * (i + 1)], 
                                         (self.__imageSize, self.__imageSize))
                    resizedFrames[i].append(resized)
                        
                count += 1
        
            transformedFrames[i] = (np.array(resizedFrames[i]) / 255.).astype(np.float16)    
        return transformedFrames
    
    def predictWebcamVideo(self, frames):
        framesArray = self.transformFramesWebcam(frames)
        
        self.__model.getLSTMmodel().load_weights("model.h5")
        transferOutput = self.__model.getTransferModel().predict(np.array(framesArray))
        
        lst = []
        lst.append(transferOutput)
        
        outputs = self.__model.getLSTMmodel().predict(np.array(lst))
        return outputs[0] > 0.5
    
    def predictVideo(self, frames):
        framesArray = self.transformFramesVideo(frames)
    
        self.__model.getLSTMmodel().load_weights("model.h5")
        lst = []
        for setOfFrames in framesArray:
            transferOutput = self.__model.getTransferModel().predict(np.array(setOfFrames))
            lst.append(transferOutput)
        
        outputs = self.__model.getLSTMmodel().predict(np.array(lst))
        
        for output in outputs:
            if output > 0.5:
                return True
        return False