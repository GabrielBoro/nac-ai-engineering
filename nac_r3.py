# EXERCICIO 3

# Faz o R2 e:

# - Realiza o processamento com imagens da webcam (executa um programa .py) O resultado esperado é uma janela da OpenCV que exibe os contornos dos 2 círculos maiores, a reta entre centros dos círculos e o valor do ângulo em relação ao plano horizontal; (máx 2 pontos)

import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
import math


def contornos(mask, frame):
    gray = frame.copy()
    gray = cv.cvtColor(gray, cv.COLOR_BGR2GRAY)
    contornos, _ = cv.findContours(mask, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

    centres = []
    for i in range(len(contornos)):
        moments = cv.moments(contornos[i])
        centres.append((int(moments['m10']/moments['m00']), int(moments['m01']/moments['m00'])))
        

    img_rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
    img_hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)

    contornos_img = frame.copy()

    image_lower_hsv = np.array([255, 0, 0])
    image_upper_hsv = np.array([255, 0, 0])

    mask_hsv = cv.inRange(img_hsv, image_lower_hsv, image_upper_hsv)

    contornos, _ = cv.findContours(mask_hsv, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE) 

    if len(centres) == 2:
        cx1=centres[0][0]
        cx2=centres[1][0]
        cy1=centres[0][1]
        cy2=centres[1][1]

        cv.drawContours(contornos_img, contornos, -1, [255, 255, 0], 5)
        cv.line(contornos_img,(cx1,cy1),(cx2, cy2),(100,255,255),5)
        m1 = (cy1 - cy2)/(cx1 - cx2)
        m2 = (cy2 - cy2)/(cx1 - cx2)
        angulo = math.atan((m2-m1)/(1-(m2*m1)))
        angulo_graus = round(math.degrees(angulo))

        font = cv.FONT_HERSHEY_SIMPLEX
        text = f'Angulo da reta: {angulo_graus}'
        textsize = cv.getTextSize(text, font, 1, 1)[0]

        textX = int((contornos_img.shape[1] - textsize[0]) / 2)
        textY = int((contornos_img.shape[0] + textsize[1]) / 2)
        cv.putText(contornos_img, text, (textX, textY), font, 1, (100, 255, 255), 1)
        return contornos_img
    
    return frame

cv.namedWindow("preview")
vc = cv.VideoCapture(0)


if vc.isOpened():
    rval, frame = vc.read()
else:
    rval = False
    
while rval:
    frame = cv.flip(frame, 1)
    cframe = frame.copy()
    cframe = cv.cvtColor(cframe,cv.COLOR_BGR2GRAY)

    circles = cv.HoughCircles(cframe,cv.HOUGH_GRADIENT,dp=5,minDist=200,param1=350,param2=250,minRadius=60,maxRadius=120)

    mask = cframe.copy()
    mask = np.zeros(cframe.shape[:2],dtype="uint8")

    if circles is not None:
        raio = []
        circles = np.uint16(np.around(circles))
        for i in circles[0,:2]:
            cv.circle(mask,(i[0],i[1]),i[2],(255,255,255),-1)
            cv.circle(frame,(i[0],i[1]),i[2],(255,255,255),5)
        frame = contornos(mask, frame).copy()

    cv.imshow("preview", frame)
    
    
    rval, frame = vc.read()
    key = cv.waitKey(20)
    if key == 27: # exit on ESC
        break

vc.release()
cv.destroyWindow("preview")