from plastic_bottle_classifier import PlasticBottleClassifier as pbc

if __name__ == '__main__':
    print("hello")
    #get dataset and stuff
    dataPath = "C:/Users/Ethan Powers/Desktop/cpts421/dev_env/data/hymenoptera_data"

    model = pbc()
    model.train(dataPath, 2, 5)
