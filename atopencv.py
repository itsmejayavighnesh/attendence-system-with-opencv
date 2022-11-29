import cv2 as jv
import numpy as np
import face_recognition as face_rec
import os
import pyttsx3 as textSpeach
from datetime import  datetime

speaker = textSpeach.init()

def resize(img, size) :
    dimension = (int(img.shape[1]*size), int(img.shape[0] * size))
    return jv.resize(img, dimension, interpolation= jv.INTER_AREA)

path = 'newfolder'
studentImg = []
studentName = []
myList = os.listdir(path)
for cl in myList :
    curimg = jv.imread(f'{path}/{cl}')
    studentImg.append(curimg)
    studentName.append(os.path.splitext(cl)[0])

def findEncoding(images) :
    imgEncodings = []
    for img in images :
        img = resize(img, 0.50)
        img = jv.cvtColor(img, jv.COLOR_BGR2RGB)
        encodeimg = face_rec.face_encodings(img)[0]
        imgEncodings.append(encodeimg)
    return imgEncodings
def MarkAttendence(name):
    with open('data.csv', 'r+') as f:
        myDatalist =  f.readlines()
        nameList = []
        for line in myDatalist :
            entry = line.split(',')
            nameList.append(entry[0])

        if name not in nameList or name in nameList:
            now = datetime.now()
            timestr = now.strftime('%H:%M')
            f.writelines(f'\n{name}, {timestr}')
            jvtranslate = name.capitalize()
            print(jvtranslate)
            statment = str('Attendence potachu  ' + jvtranslate)
            speaker.say(statment)
            speaker.runAndWait()




EncodeList = findEncoding(studentImg)

vid = jv.VideoCapture(1)
while True :
    success, frame = vid.read()
    Smaller_frames = jv.resize(frame, (0,0), None, 0.25, 0.25)

    facesInFrame = face_rec.face_locations(Smaller_frames)
    encodeFacesInFrame = face_rec.face_encodings(Smaller_frames, facesInFrame)

    for encodeFace, faceloc in zip(encodeFacesInFrame, facesInFrame) :
        matches = face_rec.compare_faces(EncodeList, encodeFace)
        facedis = face_rec.face_distance(EncodeList, encodeFace)
        #print(facedis)
        matchIndex = np.argmin(facedis)

        if matches[matchIndex] :
            name = studentName[matchIndex].upper()
            MarkAttendence(name)

    jv.imshow('video',frame)
    jv.waitKey(1)
