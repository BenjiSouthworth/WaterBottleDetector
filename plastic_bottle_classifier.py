import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision import datasets, models, transforms

class PlasticBottleClassifier:
    def __init__(self, featureExtract):
        print("created thingy")
        numClasses = 2 #bottle or not bottle
        #get the pretrained pytorch model, or now I'll use densenet
        model = models.densenet121(pretrained=True)
        #false: finetune the whole model, true: only the last layer gets updated
        if featureExtract:
            for p in model.parameters():
                p.requires_grad = False
        #change the classifier if we change the model type
        model.classifier = nn.Linear(model.classifier.in_features, numClasses)
        print(model)


    
    def train(self, trainingData, trainingLabels, testingData, testingLabels, epochs):
        print("training")
        for epoch in range(epochs):
            print("epoch "+str(epoch+1)+"/"+str(epochs))
        #call a whole bunch of other functions here

    def classifyImage(self, image):
        print("its a bottle")