# Controla um jogo da sua escolha, programa irá emular o pressionamento das teclas do teclado em função do ângulo de inclinação (ex: ângulo positivo vira para direita, ângulo negativo vira para esquerda se for um jogo de corrida). 

# Essa rubrica não pode ser feita no jupyter notebook ou google Colab. Deve ser um programa python .py. 

from pynput.keyboard import Key, Controller
import cv2 as cv
import numpy as np

def key_press(key):
    keyboard = Controller()
    if key == 'up':
        keyboard.press(Key.up)
    elif key == 'down':
        keyboard.press(Key.down)
    elif key == 'left':
        keyboard.press(Key.left)
    else:
        keyboard.press(Key.right)

def contornos(mask, frame):
    gray = frame.copy()
    gray = cv.cvtColor(gray, cv.COLOR_BGR2GRAY)
    contornos, _ = cv.findContours(mask, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

    centres = []
    for i in range(len(contornos)):
        moments = cv.moments(contornos[i])
        centres.append((int(moments['m10']/moments['m00']), int(moments['m01']/moments['m00'])))

    contornos_img = frame.copy()

    if len(centres) == 2:
        cx1=centres[0][0]
        cx2=centres[1][0]
        cy1=centres[0][1]
        cy2=centres[1][1]

        cv.drawContours(contornos_img, contornos, -1, [255, 255, 0], 5)

        (h, w) = frame.shape[:2]
        if(cy1 < h/2 and cy2 < h/2):
            key_press('up')
        if(cy1 > h/2 and cy2 > h/2):
            key_press('down')
        if(cx1 < w/2 and cx2 < w/2):
            key_press('left')
        if(cx1 > h/2 and cx2 > h/2):
            key_press('right')

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
    cframe = cv.medianBlur(cframe, 11)
    cframe = cv.cvtColor(cframe,cv.COLOR_BGR2GRAY)

    circles = cv.HoughCircles(cframe,cv.HOUGH_GRADIENT,dp=4,minDist=200,param1=300,param2=150,minRadius=60,maxRadius=120)

    mask = cframe.copy()
    mask = np.zeros(cframe.shape[:2],dtype="uint8")

    if circles is not None:
        raio = []
        circles = np.uint16(np.around(circles))
        for i in circles[0,:2]:
            cv.circle(mask,(i[0],i[1]),i[2],(255,255,255),-1)
            cv.circle(frame,(i[0],i[1]),i[2],(255,255,0),5)
        frame = contornos(mask, frame).copy()
    
    (h, w) = frame.shape[:2]
    cv.line(frame, (0, h//2), (w, h//2), (0, 255, 0), 2)
    cv.line(frame, (w//2, 0), (w//2, h), (0, 255, 0), 2)

    cv.imshow("preview", frame)
    
    
    rval, frame = vc.read()
    key = cv.waitKey(20)
    if key == 27: # exit on ESC
        break

vc.release()
cv.destroyWindow("preview")


