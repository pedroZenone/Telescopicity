# pip install opencv-python
# sudo pip install pyqtgraph

from my_Funciones import smooth
from my_Funciones import analizando_rectas
from my_Funciones import runningMeanFast

import cv2
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
from scipy.signal import medfilt

# %% Barro el video para armarme la función de desplazamiento de lamina en 
#    funcion del tiempo/metros (esto puede escalarse)

#video_str = "../Bobina_532758.asf"
#video_str = "../Videos/Bobina_532731/video.asf"
#video_str = "../Videos/Bobina_532732/video.asf"
#video_str = "../Videos/Bobina_532733/video.asf"
#video_str = "../Videos/Bobina_532734/video.asf"
video_str = "../Videos/Bobina_532735/video.asf"
#video_str = "../Videos/Bobina_532736/video.asf"
#video_str = "../Videos/Bobina_532737/video.asf"
#video_str = "../Videos/Bobina_532757/video.asf"
#video_str = "../Videos/Bobina_532758/video.asf"
#video_str = "../Videos/Bobina_532759/video.asf"
#video_str = "../Videos/Bobina_532760/video.asf"        

    
rectas_final = pd.DataFrame([])
cap = cv2.VideoCapture(video_str)    
frame_num = 0

if (cap.isOpened()== False): 
  print("Error opening video stream or file")

while(cap.isOpened()):   
    ret, frame = cap.read()
    
    if ret == True:
        frame_num =frame_num + 1
        
        rectas = analizando_rectas(frame)   
    
        if (rectas.shape[0] != 0) :
            rectas["frame"] = frame_num
            rectas_final = rectas_final.append(rectas,ignore_index = True)
            
    else:
        break

# Me fijo cuantas rayas horizontales hay      
aux = rectas_final.groupby(['frame'])['horizontal'].sum()

#aux2_bis = pd.Series(runningMeanFast(aux,20))   

# Suavizo la señal n° de rayas verticales vs frame 
aux2 = pd.Series(smooth(aux,40))
#aux3 = aux2_bis > 50  
#aux4 = aux2 > 120

# Encuentro los picos de la señal suavizada y luego derivada
from scipy.signal import find_peaks_cwt
indexes = find_peaks_cwt(np.diff(aux2), np.arange(1, 200)) # suavizo
dif_aux2 = np.diff(aux2) # derivo
ind_pico_diff_decreciente = np.argmax(dif_aux2)
ind_pico_diff_creciente = np.argmin(dif_aux2)

# El punto de start esta entre medio de la zona basal y el pico 
inic_decreciente = pd.Series(np.where( np.abs(dif_aux2) < 0.05 )[0])
inic_decreciente = np.array(inic_decreciente[inic_decreciente < ind_pico_diff_creciente])
inic_decreciente = inic_decreciente[len(inic_decreciente)-1]
start_measure = int((ind_pico_diff_creciente + inic_decreciente)/2)

#fin_decreciente = pd.Series(np.where( np.abs(dif_aux2) < 0.05 )[0])
#fin_decreciente = np.array(fin_decreciente[fin_decreciente > ind_pico_diff_decreciente])
#fin_decreciente = fin_decreciente[0]
#stop_measure = int((ind_pico_diff_decreciente + fin_decreciente)/2)

stop_measure = ind_pico_diff_decreciente

## Debug: plt.plot(aux,label = "Señal sin filtrar"),plt.plot(aux2,label = "Señal filtrada"),plt.plot(20*dif_aux2,label = "Derivada de señal filtrada"),plt.legend()

#start_measure = aux3[aux3 == False].index [0]
#start_measure = ind_pico_diff_creciente

# recorto primer tanda de falses. Como lo que quiero es detectar donde hay un cambio false - True

#index_false =aux4[aux4 == False].index
#deriv =  np.diff(index_false)
#index_stop = np.where(deriv > 20)
#stop_measure = (index_false[index_stop] + 1).values[0]
#stop_measure = 1000

# saco los datos que no van para anlizar la serie de tiempo en cada frame que detecta bordes
rectas_final = rectas_final.loc[(rectas_final.frame > start_measure) & 
                                (rectas_final.frame < stop_measure) &
                                (rectas_final.horizontal == 0 )]

# Umbralizo para descartar valores que hayan dado cualquier cosa        
rectas_final = rectas_final[(rectas_final.dist > 90) & (rectas_final.dist < 120)]    
## Debug: plt.plot(rectas_final.frame,rectas_final.dist,'*')

# Aplico filtro de mediana para quedarme con una sola recta
medians = rectas_final.groupby(by = "frame").dist.median()
## Debug: plt.plot(medians.index.values,medians)

# Aplico un filtro de mediana a la serie distancias vs frame. 
# De esta forma elimino los picos que tiene la serie
resu = medfilt(medians,7) 
## Debug: plt.plot(medians.index.values,resu)

# Con los plot comentados se puede ver como va cambiando la señal a medida que se la filtra

resu = pd.DataFrame(resu,columns = ["dist"])
resu["frame"] = medians.index.values

# %% Muestro los resultados arriba del video

from pyqtgraph.Qt import QtGui, QtCore
import pyqtgraph as pg

app = QtGui.QApplication([])
p=pg.plot()
p.setWindowTitle('Live Plot from Serial')
p.setInteractive(True)
curve = p.plot(pen=(255,0,0), name="Red X curve")

# Cargo la aplicacion que grafica en tiempo
if __name__ == '__main__':
    import sys
    if (sys.flags.interactive != 1) or not hasattr(QtCore, 'PYQT_VERSION'):
        if not QtGui.QApplication.instance():
            QtGui.QApplication.instance().exec_()

        else:
            print("App cerradando")


## Video display

fi = -np.pi*3.0/180
m0 = np.tan(fi)
b0 = int(515 - m0*380)
i = 0 

cap = cv2.VideoCapture(video_str)
    
# Check if camera opened successfully
if (cap.isOpened()== False): 
  print("Error opening video stream or file")
 
# Read until video is completed
while(cap.isOpened()):
  # Capture frame-by-frame
  ret, frame = cap.read()
  if ret == True:
 
    # Display the resulting frame
    fi = -np.pi*3.0/180
    m0 = np.tan(fi)
    b0 = int(515 - m0*380)
    x_centro = 236 # X = 236 es el centro del riel teniendo en cuenta la recta
    cv2.line(frame,(x_centro,int(m0*x_centro + b0)), (0,b0), (0,0,255),1)
    
    if (sum(resu["frame"] == i) > 0):
        x_detected = x_centro - (np.cos (fi) * resu.loc[resu.frame == i].dist)
        y_detected  = b0 + m0*x_detected
        cv2.circle(frame,(x_detected,y_detected),5,(0,0,255),2)
        
        x_new = resu.loc[resu.frame <= i].frame
        y_new = resu.loc[resu.frame <= i].dist
        curve.setData(y_new)
        
    font                   = cv2.FONT_HERSHEY_SIMPLEX
    bottomLeftCornerOfText = (200,200)
    fontScale              = 1
    fontColor              = (255,255,255)
    lineType               = 2
    
    cv2.putText(frame,str(i), 
        bottomLeftCornerOfText, 
        font, 
        fontScale,
        fontColor,
        lineType)        

    
    cv2.imshow('Frame',frame)
    i = i+1
 
    # Press Q on keyboard to  exit
    if cv2.waitKey(25) & 0xFF == ord('q'):
      break
 
  # Break the loop
  else: 
    break
 
# When everything done, release the video capture object
cap.release()
 
# Closes all the frames
cv2.destroyAllWindows()


