from models.CNN import CNN

class Main:

    def __init__(self):
        Xtrain = self.Xtrain
        Xtest = self.Xtest
        Ytrain = self.Ytrain
        Ytrain = self.Ytest


cnn = CNN()
cnn.addConvolutionalLayer()
cnn.maxPooling()
cnn.flattening()
cnn.fullConnection()
cnn.compile()
cnn.trainTest()
