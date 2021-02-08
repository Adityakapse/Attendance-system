import cv2
import numpy as np
import face_recognition

imgsachin = face_recognition.load_image_file('image basic/sachin.jpg')
imgsachin = cv2.cvtColor(imgsachin,cv2.COLOR_BGR2RGB)
imgsachintest = face_recognition.load_image_file('image basic/testsachin.jpg')
imgsachintest = cv2.cvtColor(imgsachintest,cv2.COLOR_BGR2RGB)

faceloc = face_recognition.face_locations(imgsachin)[0]
encodesachin = face_recognition.face_encodings(imgsachin)[0]
cv2.rectangle(imgsachin,(faceloc[3],faceloc[0]),(faceloc[1],faceloc[2]),(255,0,255),2)



faceloctest = face_recognition.face_locations(imgsachintest)[0]
encodesachintest = face_recognition.face_encodings(imgsachintest)[0]
cv2.rectangle(imgsachintest,(faceloctest[3],faceloctest[0]),(faceloctest[1],faceloctest[2]),(255,0,255),2)


results = face_recognition.compare_faces([encodesachin],encodesachintest)
facedis = face_recognition.face_distance([encodesachin],encodesachintest)
cv2.putText(imgsachintest,f'{results} {round(facedis[0],2)}',(50,50),cv2.FONT_HERSHEY_COMPLEX,1,(255,0,0),1)
print(results,facedis)





cv2.imshow('sachin',imgsachin)
cv2.imshow('sachintest',imgsachintest)
cv2.waitKey(1000)