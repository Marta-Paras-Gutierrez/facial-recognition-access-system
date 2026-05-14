# Librerías empleadas
import cv2				# Utilizar OpenCV para reconocimiento de rostros
import os				# Para crear, abrir o trabajar con directorios
import numpy as np		# Para el cálculo numérico y el análisis de los datos

# Creación de la carpeta para el almacenamiento de los archivos
if not os.path.exists('Archivos'):
    print('Se ha creado la carpeta: Archivos')
    os.makedirs('Archivos')

# Grabar el vídeo con el que se trabajará
capturando = cv2.VideoCapture(0,cv2.CAP_DSHOW)   # DSHOW para que ocupe toda la ventana de visualización

# Guardar el vídeo en la carpeta y formato establecidos
salida = cv2.VideoWriter("G:/TFG/Programa/Marta_5.mp4", cv2.VideoWriter_fourcc(*'mp4v'), 10.0, (640, 480))	

while True:
	ret,frame = capturando.read()
	frame = cv2.flip(frame,1)	# Para visualizar en modo espejo
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	recuadroAuxiliar = frame.copy()
	
	cv2.imshow('Ventana de visualizacion',frame)
	salida.write(frame)

	k = cv2.waitKey(1)
	if k == ord('q'):		# Establecer como cerrar el vídeo
		break

capturando.release()
salida.release()
cv2.destroyAllWindows()