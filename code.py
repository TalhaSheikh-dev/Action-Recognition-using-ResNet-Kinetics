
# import the necessary packages
import numpy as np
import argparse
import imutils
import sys
import cv2


model = "resnet-34_kinetics.onnx"
classess = "action_recognition_kinetics.txt"
input_video = "Practicing_hurdles.webm" 

CLASSES = open(classess).read().strip().split("\n")
SAMPLE_DURATION = 16
SAMPLE_SIZE = 112

# load the human activity recognition model
print("[INFO] loading human activity recognition model...")
net = cv2.dnn.readNet(model)
# grab a pointer to the input video stream
print("[INFO] accessing video stream...")
vs = cv2.VideoCapture(input_video if input_video else 0)


# loop until we explicitly break from it
while True:
	# initialize the batch of frames that will be passed through the
	# model
	frames = []
	# loop over the number of required sample frames
	for i in range(0, SAMPLE_DURATION):
		# read a frame from the video stream
		(grabbed, frame) = vs.read()
		if not grabbed:
			break
		frame = imutils.resize(frame, width=400)
		frames.append(frame)
	if len(frames) == 0:
		break
	# now that our frames array is filled we can construct our blob
	blob = cv2.dnn.blobFromImages(frames, 1.0,
		(SAMPLE_SIZE, SAMPLE_SIZE), (114.7748, 107.7354, 99.4750),
		swapRB=True, crop=True)
	blob = np.transpose(blob, (1, 0, 2, 3))
	blob = np.expand_dims(blob, axis=0)
    
	# pass the blob through the network to obtain our human activity
	# recognition predictions
	net.setInput(blob)
	outputs = net.forward()
	label = CLASSES[np.argmax(outputs)]
       
	# loop over our frames
	for frame in frames:
		# draw the predicted activity on the frame
		cv2.rectangle(frame, (0, 0), (300, 40), (0, 0, 0), -1)
		cv2.putText(frame, label, (10, 25), cv2.FONT_HERSHEY_SIMPLEX,
			0.8, (255, 255, 255), 2)
		# display the frame to our screen
		cv2.imshow("Activity Recognition", frame)
		key = cv2.waitKey(1) & 0xFF
		# if the `q` key was pressed, break from the loop
		if key == ord("q"):
			break
vs.release()
cv2.destroyWindow("Activity Recognition")