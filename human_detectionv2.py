import cv2 as cv
import argparse
import sys
import numpy as np
import os.path
from datetime import datetime

# Initialize the parameters
confThreshold = 0.3  # Confidence threshold
nmsThreshold = 0.2  # Non-maximum suppression threshold
inpWidth = 1660  # Width of network's input image
inpHeight = 925  # Height of network's input image

parser = argparse.ArgumentParser(description='Object Detection using YOLO in OPENCV')
parser.add_argument('--image', help='Path to image file.')
parser.add_argument('--video', help='Path to video file.')
parser.add_argument('--path', help='image path')
parser.add_argument('--filename', help='filename')

args = parser.parse_args()

#Get input arguments --path & --filename without file extensions
inputFile = ( args.path + args.filename + ".jpg" )
outputFile = ( args.path + args.filename + ".png" )

classesFile = "human.txt" #file including object name
classes = None
with open(classesFile, 'rt') as f:
    classes = f.read().rstrip('\n').split('\n')

# Give the configuration and weight files for the model and load the network using them.
modelConfiguration = "human.cfg" #yolo modelfile
modelWeights = "human.weights" #yolo weightfile
net = cv.dnn.readNetFromDarknet(modelConfiguration, modelWeights)
net.setPreferableBackend(cv.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv.dnn.DNN_TARGET_CPU)

def getOutputsNames(net):
    # Get the names of all the layers in the network
    layersNames = net.getLayerNames()
    # Get the names of the output layers, i.e. the layers with unconnected outputs
    return [layersNames[i[0] - 1] for i in net.getUnconnectedOutLayers()]
# Draw the predicted bounding box of object
def drawPred(classId, conf, left, top, right, bottom):
    # Draw a bounding box.
    cv.rectangle(frame, (left, top), (right, bottom), (255, 178, 50), 2)
    label = '%.2f' % conf
    # Get the label for the class name and its confidence
    if classes:
        assert (classId < len(classes))
        label = '%s:%s' % (classes[classId], label)

    # Display the label at the top of the bounding box
    labelSize, baseLine = cv.getTextSize(label, cv.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    top = max(top, labelSize[1])
    #cv.putText(frame, label, (left, top + 30), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)



# Remove the bounding boxes with low confidence using non-maxima suppression

def color_filter(img, r, g, b):
    colors = [b, g, r]
    result = np.zeros(img.shape, dtype=np.uint8)
    for i in range(3):
        result[:, :, i] = np.where(img[:, :, i] < colors[i], 0, 255)
    return result.astype(np.uint8)


def postprocess(frame, outs):
    original_frame = frame.copy()
    extraction_image = np.zeros([frame.shape[0], frame.shape[1], 3], dtype=np.uint8)
    extraction_image.fill(255)

    frameHeight = frame.shape[0]
    frameWidth = frame.shape[1]

    # Scan through all the bounding boxes output from the network and keep only the
    # ones with high confidence scores. Assign the box's class label as the class with the highest score.
    classIds = []
    confidences = []
    boxes = []
    for out in outs:
        for detection in out:
            scores = detection[5:]
            classId = np.argmax(scores)
            confidence = scores[classId]
            if confidence > confThreshold:
                center_x = int(detection[0] * frameWidth)
                center_y = int(detection[1] * frameHeight)
                width = int(detection[2] * frameWidth)
                height = int(detection[3] * frameHeight)
                left = int(center_x - width / 2)
                top = int(center_y - height / 2)

                classIds.append(classId)
                confidences.append(float(confidence))
                boxes.append([left, top, width, height])

    # Perform non maximum suppression to eliminate redundant overlapping boxes with
    # lower confidences.
    indices = cv.dnn.NMSBoxes(boxes, confidences, confThreshold, nmsThreshold)
    for i in indices:
        i = i[0]
        box = boxes[i]
        left = box[0]
        top = box[1]
        width = box[2]
        height = box[3]
        index = classIds[i]
        if (classIds[i] == 0 and width < 500):
            crop_image = original_frame[top:top+height, left:left + width]
            extraction_image[top:top + height, left:left + width] = crop_image

            drawPred(classIds[i], confidences[i], left, top, left + width, top + height)

    low_green = np.array([25, 52, 72])
    high_green = np.array([102, 255, 255])

    imgHSV = cv.cvtColor(extraction_image, cv.COLOR_BGR2HSV)
    
    # create the Mask
    mask = cv.inRange(imgHSV, low_green, high_green)
    
    # inverse mask
    mask = 255 - mask

    res = cv.bitwise_and(extraction_image, extraction_image, mask=mask)

    res[mask == 0] = (255, 255, 255)

    #res = cv.resize(res, (int(res.shape[1] / 2), int(res.shape[0] / 2)))

    h, w, c = res.shape
    # append Alpha channel -- required for BGRA (Blue, Green, Red, Alpha)
    image_bgra = np.concatenate([res, np.full((h, w, 1), 255, dtype=np.uint8)], axis=-1)
    
    # create a mask where white pixels ([255, 255, 255]) are True
    white = np.all(res == [255, 255, 255], axis=-1)
    
    # change the values of Alpha to 0 for all the white pixels
    image_bgra[white, -1] = 0
    
    # save the image
    cv.imwrite(outputFile, image_bgra)

# Process inputs
frame = cv.imread(inputFile);

# Create a 4D blob from a frame.
blob = cv.dnn.blobFromImage(frame, 1 / 255, (inpWidth, inpHeight), [0, 0, 0], 1, crop=False)

# Sets the input to the network
net.setInput(blob)

# Runs the forward pass to get output of the output layers
outs = net.forward(getOutputsNames(net))

# Remove the bounding boxes with low confidence
postprocess(frame, outs)
