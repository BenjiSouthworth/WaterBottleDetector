import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision import datasets, models, transforms
import os
import time

class PlasticBottleClassifier:
    def __init__(self):
        print("created thingy")
        self.numClasses = 2 #bottle or not bottle
        #get the pretrained pytorch model, or now I'll use densenet
        self.model = models.densenet121(pretrained=True)
        self.inputSize = 224
        #finetune only the last layer
        for p in self.model.parameters():
            p.requires_grad = False
        #change the classifier if we change the model type
        self.model.classifier = nn.Linear(self.model.classifier.in_features, self.numClasses)
        #print(model)

    def train(self, dataPath, epochs, batchSize):
        print("testing")
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        if torch.cuda.is_available():
            print("using gpu")
        else:
            print("using cpu")
        dataloader = self.loadData(dataPath, batchSize)
        criterion = nn.CrossEntropyLoss() #choose the loss function (how we determine how far off our model is from the goal)
        #choose parameters that will be updated
        parameters = []
        for name,param in self.model.named_parameters():
            if param.requires_grad == True:
                parameters.append(param)
                print("\t",name)
        optimizer = optim.SGD(parameters, lr=0.001, momentum=0.9) #choose the algorithm that will be used to update parameters during training

        totalTime = 0.0
        for epoch in range(epochs):
            print("epoch "+str(epoch+1)+"/"+str(epochs))
            timer = time.time()

            for phase in ['train', 'val']:
                if phase == 'train':
                    self.model.train()
                else:
                    self.model.eval()

                runningLoss = 0.0
                runningCorrects = 0

                for inputs, labels in dataloader[phase]:
                    inputs = inputs.to(device)
                    labels = labels.to(device)

                    optimizer.zero_grad() #initialize the optimizer gradients to 0
                    with torch.set_grad_enabled(phase == 'train'):
                        outputs = self.model(inputs)
                        loss = criterion(outputs, labels)

                    _, preds = torch.max(outputs, 1)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                    # statistics
                    runningLoss += loss.item() * inputs.size(0)
                    runningCorrects += torch.sum(preds == labels.data)

                loss = runningLoss / len(dataloader[phase].dataset)
                accuracy = runningCorrects.double() / len(dataloader[phase].dataset)

                #print("loss: "+str(loss)+" accuracy: "+str(accuracy))
                print('{} loss: {:.4f} accuracy: {:.4f}'.format(phase, loss, accuracy))

            elapsed = time.time() - timer
            totalTime += elapsed

        print("total time elapsed: "+str(totalTime))



    def classifyImage(self, image):
        print("its a bottle")

    def loadData(self, dataPath, batchSize):
        dataTransforms = {
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
        dataset = {x: datasets.ImageFolder(os.path.join(dataPath, x), dataTransforms[x]) for x in ['train', 'val']}
        dataloader = {x: torch.utils.data.DataLoader(dataset[x], batch_size=batchSize, shuffle=True, num_workers=3) for x in ['train', 'val']}
        return dataloader