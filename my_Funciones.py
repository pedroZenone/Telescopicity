#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 21 13:38:46 2018

@author: pedrp
"""
import cv2
import numpy as np
import numpy
from matplotlib import pyplot as plt
import pandas as pd
from scipy.signal import medfilt
import pdb

#  @ func: Detecta el borde de la lamina y le deja un cirulo
#  @param frame: imagen a procesar. 
#  @return: dataframa que contiene pendiente y ordenada al origen de las
#           rectas que bordean la lamina. Tambien devuelve la distancia desde
#           el centro de la cinta a la recta y el punto desde el cual se medio
#           al centro de la cinta

def analizando_rectas(frame):
    
#    cap = cv2.VideoCapture("./Videos/Bobina_532730/video1/video1.asf")    
#    
#    for i in range(frame_num):    
#        ret, frame = cap.read()
    
    # Recta patron para medir distancia al punto medio
    fi = -np.pi*3.0/180  # la muevo 3 grados porque la figura viene rotada
    m0 = np.tan(fi)
    b0 = int(515 - m0*380)
    x_centro = 236 # X = 236 es el centro del riel teniendo en cuenta la recta
    cv2.line(frame,(x_centro,int(m0*x_centro + b0)), (0,b0), (0,0,255),1)
    
    # recorto la imagen y solo analizo la ultima parte de la cinta. Esta zona es mas lineal
    offset = 350
    frame_ = frame[offset:,:,:] 
    
    # Paso a otra escala de colores: HSV
    hsv = cv2.cvtColor(frame_, cv2.COLOR_BGR2HSV) #convert it to hsv
    h, s, v = cv2.split(hsv)
    v -= 60  # le saco brillo ya que las barras de la cinta resplandecen mucho
    v[v<0] = 0
    img = cv2.merge((h, s, v))
    
    # saco ruido a la imagen        
    blur = cv2.GaussianBlur(img,(19,19),0)
    
    # filtro detección de bordes
    edges = cv2.Canny(blur,5,40)
    
    # transformada de Hough. Paso al dominio de las rectas. 
    # Los puntos que esten contenidos en una misma recta van a ser detectados or Hough
    lines = cv2.HoughLines(edges,1,np.pi/180, 50)
    test = []
    horizontal = 0
    
    if (lines is not None):
                for i in range(lines.shape[0]):
                    for r,theta in lines[i]:
                        
                        if((theta < 1) & (theta != 0)):                     
                            a = np.cos(theta); b = np.sin(theta)     
                            
                            x1 = int(a*r  + 1000*(-b))              
                            y1 = int( b*r + 1000*(a)) + offset             
                
                            x2 = int(a*r - 1000*(-b))   
                            y2 = int( b*r - 1000*(a)) + offset 
#                            pdb.set_trace() 
                            m = (y1 - y2)/(x1 - x2)
                            b = y1 - (m*x1)
                            
                            test.append({"x1":x1,"x2":x2, "m": m, "b":b, "horizontal": 0})                        
                            cv2.line(frame,(x1,y1), (x2,y2), (0,0,255),1)
                            
                        if((theta >= 1)):
                            test.append({"x1":0,"x2":0, "m": 0, "b":0, "horizontal": 1})
                            
                            
    rectas = pd.DataFrame(test)  
                      
    if (len(test) != 0):
        
        rectas["interseccion"] = (rectas["b"] - b0)/(m0 - rectas["m"])
        rectas["dist"] = (x_centro - rectas["interseccion"] )/ np.cos(fi)        
        
    return rectas        

######################################################################################

def smooth(x,window_len=11,window='hanning'):
    """smooth the data using a window with requested size.
    
    This method is based on the convolution of a scaled window with the signal.
    The signal is prepared by introducing reflected copies of the signal 
    (with the window size) in both ends so that transient parts are minimized
    in the begining and end part of the output signal.
    
    input:
        x: the input signal 
        window_len: the dimension of the smoothing window; should be an odd integer
        window: the type of window from 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'
            flat window will produce a moving average smoothing.

    output:
        the smoothed signal
        
    example:

    t=linspace(-2,2,0.1)
    x=sin(t)+randn(len(t))*0.1
    y=smooth(x)
    
    see also: 
    
    numpy.hanning, numpy.hamming, numpy.bartlett, numpy.blackman, numpy.convolve
    scipy.signal.lfilter
 
    TODO: the window parameter could be the window itself if an array instead of a string
    NOTE: length(output) != length(input), to correct this: return y[(window_len/2-1):-(window_len/2)] instead of just y.
    """

    if x.ndim != 1:
        raise (ValueError, "smooth only accepts 1 dimension arrays.")

    if x.size < window_len:
        raise (ValueError, "Input vector needs to be bigger than window size.")


    if window_len<3:
        return x


    if not window in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']:
        raise (ValueError, "Window is on of 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'")


    s=np.r_[x[window_len-1:0:-1],x,x[-2:-window_len-1:-1]]
    #print(len(s))
    if window == 'flat': #moving average
        w=np.ones(window_len,'d')
    else:
        w=eval('numpy.'+window+'(window_len)')

    y=np.convolve(w/w.sum(),s,mode='valid')
    return y

#####################################

#  @ func: Suaviza la señal con un filtro de media movil
#  @param x: señal a procesar
#  @param N: largo del filtro
#  @return: señal suavizada

def runningMeanFast(x, N):
    return np.convolve(x, np.ones((N,))/N)[(N-1):]