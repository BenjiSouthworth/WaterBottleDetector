import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision import datasets, models, transforms
import os

class PlasticBottleClassifier:
    def __init__(self, featureExtract):
        print("created thingy")
        numClasses = 2 #bottle or not bottle
        #get the pretrained pytorch model, or now I'll use densenet
        model = models.densenet121(pretrained=True)
        inputSize = 224
        #false: finetune the whole model, true: only the last layer gets updated
        if featureExtract:
            for p in model.parameters():
                p.requires_grad = False
        #change the classifier if we change the model type
        model.classifier = nn.Linear(model.classifier.in_features, numClasses)
        print(model)

    def train(self, dataPath, epochs, batchSize):
        print("testing")
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        if torch.cuda.is_available():
            print("using gpu")
        else:
            print("using cpu")
        data = self.loadData(dataPath)
        for epoch in range(epochs):
            print("epoch "+str(epoch+1)+"/"+str(epochs))


    def classifyImage(self, image):
        print("its a bottle")

    def loadData(self, dataPath):
        data_transforms = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(self.inputSize),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize(self.inputSize),
            transforms.CenterCrop(self.inputSize),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        }
        dataset = {x: datasets.ImageFolder(os.path.join(dataPath, x), data_transforms[x]) for x in ['train', 'val']}
        dataloaders_dict = {x: torch.utils.data.DataLoader(dataset[x], batch_size=self.batchSize, shuffle=True, num_workers=3) for x in ['train', 'val']}
        return dataloaders_dict