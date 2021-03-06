
# import the necessary packages
from pyimagesearch import config
from torchvision import models
import numpy as np
import torch
import cv2

def preprocess_image(image):
	# swap the color channels from BGR to RGB, resize it, and scale
	# the pixel values to [0, 1] range
	image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
	image = cv2.resize(image, (config.IMAGE_SIZE, config.IMAGE_SIZE))
	image = image.astype("float32") / 255.0

	# subtract ImageNet mean, divide by ImageNet standard deviation,
	# set "channels first" ordering, and add a batch dimension
	image -= config.MEAN
	image /= config.STD
	image = np.transpose(image, (2, 0, 1))
	image = np.expand_dims(image, 0)

	# return the preprocessed image
	return image

import PySimpleGUI as sg
  
# Add some color
sg.theme('SandyBeach')     
  
# Very basic window.
layout = [
    [sg.Text('Please enter your desired image for classification')],
    [sg.Text('Image.type', size =(15, 1)), sg.InputText()],
    [sg.Submit()]
]
  
window = sg.Window('Simple data entry window', layout)
event, value = window.read()
window.close()

source = "images/"
inn = value[0]#input()
val = source+inn

image_input = val

# load our the network weights from disk, flash it to the current
# device, and set it to evaluation mode
print("[INFO] loading {}...".format("resnet"))
model = models.resnet50(pretrained=True).to(config.DEVICE)
model.eval()

# load the image from disk, clone it (so we can draw on it later),
# and preprocess it
print("[INFO] loading image...")
image = cv2.imread(image_input)
orig = image.copy()
image = preprocess_image(image)

# convert the preprocessed image to a torch tensor and flash it to
# the current device
image = torch.from_numpy(image)
image = image.to(config.DEVICE)

# load the preprocessed the ImageNet labels
print("[INFO] loading ImageNet labels...")
imagenetLabels = dict(enumerate(open(config.IN_LABELS)))

# classify the image and extract the predictions
print("[INFO] classifying image with '{}'...".format("resnet"))
logits = model(image)
probabilities = torch.nn.Softmax(dim=-1)(logits)
sortedProba = torch.argsort(probabilities, dim=-1, descending=True)


#arrays to hold th top stat
topclasses = [' ']*100
topprobs   = [' ']*100

for (i, idx) in enumerate(sortedProba[0, :5]):
	topclasses[i] = imagenetLabels[idx.item()].strip()
	topprobs[i]   = str("{:.2f}%".format(probabilities[0, idx.item()] * 100))
	#print("{}. {}: {:.2f}%".format
		#(i, imagenetLabels[idx.item()].strip(),
		#probabilities[0, idx.item()] * 100))
#^^print functionality for non-gui^^


sg.theme('DarkGreen')     
  
# Very basic window.

layout = [
    [sg.Text('Top Classication Result')],
    [sg.Text(topclasses[0], size =(0, 1)),sg.Text(': ', size =(0, 1)) ,sg.Text(topprobs[0], size =(10, 1))],
    [sg.Button('View Image and watermark'),sg.Button('Exit')]
]
  
window = sg.Window('RESULTS', layout)

while True:             # Event Loop
		event, value = window.read()
		print(event, value)
		if event == sg.WIN_CLOSED or event == 'Exit':
			break
		if event == 'View Image and watermark':            #draw it
			(label, prob) = (imagenetLabels[probabilities.argmax().item()],
				probabilities.max().item())
			cv2.putText(orig, "Label: {}, {:.2f}%".format(label.strip(), prob * 100),
				(10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
			cv2.imshow("Classification", orig)
			cv2.waitKey(0)
window.close()


#exits here woooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooo
