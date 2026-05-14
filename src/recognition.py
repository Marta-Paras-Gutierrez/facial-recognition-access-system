# Librerías empleadas
import cv2				# Utilizar OpenCV para reconocimiento de rostros
import os				# Para crear, abrir o trabajar con directorios
import numpy as np		# Para el cálculo numérico y el análisis de los datos
import time				# Para insertar tiempos de espera

# Registro de personas que usaron el entrenamiento de reconocimiento
carpetaDeTrabajo = 'G:/TFG/Programa/Archivos/Rostros'
listaPersonal = os.listdir(carpetaDeTrabajo)
print('Lista de personas: ', listaPersonal)


# Escoger que método se va a usar para el reconocimiento
metodo = int(input("¿Qué método quieres usar? 1-Eigen   2-Fisher   3-LBPH\n"))
valor_deteccion = 0
print("Método elegido = ", metodo)

# Leyendo el modelo --> Leer el archivo con el mismo nombre que el modelo de los rostros
if metodo == 1:
	reconocimiento_facial = cv2.face.EigenFaceRecognizer_create()
	reconocimiento_facial.read('modeloReconocido-EigenFace.xml')
	valor_deteccion = 5500			# Vídeos grabados 2500 y cámara 5500
	print("Empleando el método 1-Eigen")

elif metodo == 2:
	reconocimiento_facial = cv2.face.FisherFaceRecognizer_create()
	reconocimiento_facial.read('modeloReconocido-FisherFace.xml')
	valor_deteccion = 5				# Vídeos grabados 3 y cámara 10
	print("Empleando el método 2-Fisher")

elif metodo == 3:
	reconocimiento_facial = cv2.face.LBPHFaceRecognizer_create()
	reconocimiento_facial.read('modeloReconocido-LBPHFace.xml')
	valor_deteccion = 65			# Vídeos grabados 50 y cámara 70
	print("Empleando el método 3-LBPH")

else: print("Error")

# Usar vídeos para ver si hace bien el reconocimiento
#capturando = cv2.VideoCapture('G:/TFG/Programa/Archivos/Marta.mp4')
capturando = cv2.VideoCapture(0, cv2.CAP_DSHOW)	# Usar la cámara del ordenador
#capturando = cv2.VideoCapture(url)		# Usar la cámara integrada del ESP32

faceClassif = cv2.CascadeClassifier('Encuadre_de_las_caras.xml')

while True:
	ret,frame = capturando.read()
	if ret == False: 
		break
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	recuadroAuxiliar = gray.copy()

	faces = faceClassif.detectMultiScale(gray, 1.3, 5)

	for (x,y,w,h) in faces:
		cv2.rectangle(frame, (x,y),(x+w,y+h),(0,255,0),2)
		rostro = recuadroAuxiliar[y:y+h, x:x+w]
        # Usar la misma escala que en el reconocedor de rostros
		rostro = cv2.resize(rostro,(150,150), interpolation=cv2.INTER_CUBIC)

        # Predecir una etiqueta y la distancia para una imagen de entrada determinada
		resultado = reconocimiento_facial.predict(rostro)
        # Visualizar los dos valores obtenidos
		cv2.putText(frame, '{}'.format(resultado), (x, y-5), 1, 1.3, (255,255,0), 1, cv2.LINE_AA)
		
        # Indicar de quien es el rostro según el entrenamiento
		if resultado[1] < valor_deteccion:  # Valores cercanos a 0 son los que son reconocidos
			cv2.putText(frame, '{}'.format(listaPersonal[resultado[0]]), (x, y-30), 2, 1.1, (203,192,203), 1, cv2.LINE_AA)
			cv2.rectangle(frame, (x,y),(x+w,y+h),(0,255,0),2)
			print("\033[32mRostro detectado {}\033[0m".format(listaPersonal[resultado[0]]))

		else:
			cv2.putText(frame, 'Desconocido', (x, y-30), 2, 0.8, (0,0,255), 1, cv2.LINE_AA)
			cv2.rectangle(frame, (x,y),(x+w,y+h),(0,255,255),2)
			print("\033[31mPersona desconocida\033[0m")

	cv2.imshow('Ventana de visualizacion',frame)
	
	k = cv2.waitKey(1)
	if k == 27:		# Establecer como cerrar el vídeo, tecla esc
		break

capturando.release()
cv2.destroyAllWindows()