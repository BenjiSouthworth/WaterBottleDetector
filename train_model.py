from plastic_bottle_classifier import PlasticBottleClassifier as pbc

if __name__ == '__main__':
    print("hello")
    #get dataset and stuff
    trainingData = []
    trainingLabels = []
    testingData = []
    testingLabels = []

    model = pbc(featureExtract=True)
    model.train(trainingData, trainingLabels, testingData, testingLabels, 420)
