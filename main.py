import cv2
import numpy as np
import math
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import time
import os

try:
	cap=cv2.VideoCapture(0)
except:
	print("Can't open Camera. Camera engaged or not available.")

model=load_model('prediction_model.h5')
face_cascade = cv2.CascadeClassifier('haar_face.xml')

while(1):
	ret, frame=cap.read()
	frame=cv2.flip(frame, 1)
	k = cv2.waitKey(1)
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	faces = face_cascade.detectMultiScale(gray, 1.3, 5)
	roi=frame[100:300, 100:300]

	for (x,y,w,h) in faces:
		roi=frame[x:x+w, y:y+h]
		cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)

	# img=cv2.cvtColor(roi, cv2.COLOR_GRAY2RGB)
	img=cv2.resize(roi, (48, 48))
	x=image.img_to_array(img)
	x=np.expand_dims(x, axis=0)
	x=x/255
	images=np.vstack([x])
	arr=model.predict(images)
	gesture=np.argmax(arr)
	dictionary={
	    0: 'angry',
	    1: 'happy',
	    2: 'neutral',
	    3: 'sad',
	    4: 'surprise'
	}

	answer=dictionary[gesture]

	k = cv2.waitKey(5) & 0xFF
	if(answer=="happy" and k==32):
		os.system("firefox https://www.youtube.com/watch?v=WsptdUFthWI")

	if(answer=="sad" and k==32):
		os.system("firefox https://www.youtube.com/watch?v=hoNb6HuNmU0")

			# cv2.rectangle(frame,(c1, r1),(c2, r2),(0,255,0),0) 

	font=cv2.FONT_HERSHEY_SIMPLEX
	cv2.putText(frame, answer, (50, 50), font, 1, (0, 255, 255), 2, cv2.LINE_4)
	cv2.imshow('frame',frame)
	cv2.imshow('roi', roi)

		# cv2.imshow('frame',frame)

	if k == 27:
		break
