import cv2
import numpy as np
import face_recognition 

imgElon = face_recognition.load_image_file("ImagesBasic/elonmusk.jpg")
imgElon = cv2.cvtColor(imgElon, cv2.COLOR_BGR2RGB)
imgElonTest = face_recognition.load_image_file("ImagesBasic/elonmusk_test.jpg")
imgElonTest = cv2.cvtColor(imgElonTest, cv2.COLOR_BGR2RGB)

faceloc = face_recognition.face_locations(imgElon)[0]
encodeElon = face_recognition.face_encodings(imgElon)[0]
cv2.rectangle(imgElon, (faceloc[3], faceloc[0]), (faceloc[1], faceloc[2]), (255, 0, 255), 2)

facelocTest = face_recognition.face_locations(imgElonTest)[0]
encodeElonTest = face_recognition.face_encodings(imgElonTest)[0]
cv2.rectangle(imgElonTest, (facelocTest[3], facelocTest[0]), (facelocTest[1], facelocTest[2]), (255, 0, 255), 2)

results = face_recognition.compare_faces([encodeElon], encodeElonTest)
faceDis = face_recognition.face_distance([encodeElon], encodeElonTest)
print(results, faceDis )
cv2.putText(imgElonTest, f'{results} {round(faceDis[0], 2)}', (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)

cv2.imshow("Elon Musk", imgElon)
cv2.imshow("Elon Musk Test", imgElonTest)
cv2.waitKey(0)