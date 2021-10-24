from plastic_bottle_classifier import PlasticBottleClassifier as pbc

if __name__ == '__main__':
    print("hello")
    #get dataset and stuff
    dataPath = "somewhere"

    model = pbc(featureExtract=True)
    model.train(dataPath, 10, 5)
