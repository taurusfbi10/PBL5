import cv2
import numpy as np
detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml');
cap=cv2.VideoCapture(0);
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read('trainner/trainner.yml')
id=0
font = cv2.FONT_HERSHEY_SIMPLEX
while(True):
    ret,img=cap.read();
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY);
    faces = detector.detectMultiScale(gray, 1.3, 5);
    for (x,y,w,h) in faces:
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
        id, conf = recognizer.predict(gray[y:y+h,x:x+w])
        if id==1:
            id="1"
        elif (id==2):
            id="2"
        cv2.putText(img,str(id), (x,y),cv2.FONT_HERSHEY_SIMPLEX, 1, 255)
    cv2.imshow('frame',img);
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break;
cap.release();
cv2.destroyAllWindows();
   
