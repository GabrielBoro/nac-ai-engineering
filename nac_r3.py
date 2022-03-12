# EXERCICIO 3
# Faz o R2 e:

# - Realiza o processamento com imagens da webcam (executa um programa .py) O resultado esperado é uma janela da OpenCV que exibe os contornos dos 2 círculos maiores, a reta entre centros dos círculos e o valor do ângulo em relação ao plano horizontal; (máx 2 pontos)

# - Essa rubrica não pode ser feita no jupyter notebook ou google Colab. Deve ser um programa python .py. 

import cv2 as cv
import numpy as np

def contornos(mask):
    contornos, _ = cv.findContours(mask, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

    centres = []
    for i in range(len(contornos)):
        moments = cv.moments(contornos[i])
        centres.append((int(moments['m10']/moments['m00']), int(moments['m01']/moments['m00'])))
        
    mask = cv.cvtColor(mask,cv.COLOR_GRAY2BGR)

    img_rgb = cv.cvtColor(mask, cv.COLOR_BGR2RGB)
    img_hsv = cv.cvtColor(mask, cv.COLOR_BGR2HSV)

    contornos_img = mask.copy()

    image_lower_hsv = np.array([10, 10, 10])
    image_upper_hsv = np.array([255, 255, 255])

    mask_hsv = cv.inRange(img_hsv, image_lower_hsv, image_upper_hsv)

    contornos, _ = cv.findContours(mask_hsv, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE) 

    if len(centres) == 2:
        cx1=centres[0][0]
        cx2=centres[1][0]
        cy1=centres[0][1]
        cy2=centres[1][1]
        cv.drawContours(contornos_img, contornos, -1, [255, 0, 0], 5)
        cv.line(contornos_img,(cx1,cy1),(cx2, cy2),(255,0,0),5)

        return contornos_img
    
    return mask

cv.namedWindow("preview")
vc = cv.VideoCapture(0)


if vc.isOpened(): # try to get the first frame
    rval, frame = vc.read()
else:
    rval = False
    
while rval:
    cframe = frame.copy()
    cframe = cv.cvtColor(cframe,cv.COLOR_BGR2GRAY)

    # cframe = cv.medianBlur(cframe,5)
#     image: imagem de entrada na escala de ciza. 
# -     method: Define o metódo de detecção de circulos.
# -     dp: relação entre o tamanho da imagem e o tamanho do acumulador. Um dp grande "pega" bordas mais tênues.
# - minDist: Distância minima entre centros (x,y) dos circulos detectados
# - param1: Valor do gradiente usado para lidar com a detecção de bordas
# - param2: Limiar do Acumulador usado pelo metódo. Se muito baixo, retorna mais circulos (incluindo círculos falsos). Se mais alto, mais círculos serão potencialmente retornados.
# - minRadius: Raio minimo (em pixels).
# - maxRadius: Raio máximo (em pixels).
    circles = cv.HoughCircles(cframe,cv.HOUGH_GRADIENT,dp=5,minDist=200,param1=200,param2=200,minRadius=50,maxRadius=80)

    mask = cframe.copy()

    if circles is not None:
        raio = []
        circles = np.uint16(np.around(circles))
        for i in circles[0,:2]:
            cv.circle(mask,(i[0],i[1]),i[2],(255,255,255),-1)
        mask = contornos(mask).copy()

    mask = cv.flip(mask, 1)
    cv.imshow("preview", mask)
    
    
    rval, frame = vc.read()
    key = cv.waitKey(20)
    if key == 27: # exit on ESC
        break

vc.release()
cv.destroyWindow("preview")