# Librerías empleadas
import cv2				# Utilizar OpenCV para reconocimiento de rostros
import os				# Para crear, abrir o trabajar con directorios
import numpy as np		# Para el cálculo numérico y el análisis de los datos

# Registro de a quien se va a reconocer
nombreDeLaPersona = 'Claudia'
carpetaGuardarRostros = 'G:/TFG/Programa/Rostros'
carpetaPersonal = carpetaGuardarRostros + '/' + nombreDeLaPersona

# Si la carpeta donde se van a guardar los rostros no está creada, se creará
if not os.path.exists(carpetaPersonal):
    print('Carpeta creada: ', carpetaPersonal)
    os.makedirs(carpetaPersonal)

# Ejecutar el detector de rostros
faceClassif = cv2.CascadeClassifier('Encuadre_de_las_caras.xml')

# Vídeo con el que voy a trabajar
capturando = cv2.VideoCapture('{}.mp4'.format(nombreDeLaPersona))
# Para registrar el número de rostros 
cuenta_rostros = 0	# si uso mas de un vídeo hay que cambiar este valor

while True:
	ret,marco = capturando.read()   # Trabajar sobre el vídeo
	if ret == False: 
		print("Error")
		break

	marco = cv2.resize(marco, (640,480))	# Redimensionar por si el vídeo es muy grande
	gray = cv2.cvtColor(marco, cv2.COLOR_BGR2GRAY)  # Convertirlo a escala de grises para trabajar con ello
	recuadroAuxiliar = marco.copy()

    # Activación del reconocimiento facial
	faces = faceClassif.detectMultiScale(gray, 1.4, 5)

	for (x,y,w,h) in faces:
		print(f'Guardando rostros... {cuenta_rostros} / 900')	# Para ver si hay algún error durante el proceso

		cv2.rectangle(marco, (x,y),(x+w,y+h),(0,255,0),2)	# Poner el rectángulo en el vídeo
		rostro = recuadroAuxiliar[y:y+h, x:x+w]				# Posición del recuadro de reconocimiento de rostros
		rostro = cv2.resize(rostro,(150,150), interpolation=cv2.INTER_CUBIC)		# Establecer el tamaño del recorte de la cara

		cv2.imwrite(carpetaPersonal + '/rostro_{}.jpg'.format(cuenta_rostros), rostro)
		cuenta_rostros += 1

	cv2.imshow('Ventana de visualizacion',marco)
	
	# Establecer como cerrar el vídeo
	k = cv2.waitKey(1)
	if k == 27 or cuenta_rostros >= 900:		# Si uso mas de un vídeo hay que modificar el valor final de fotos
		print("Acabada la recogida de rostros")	# Poner mensaje por pantalla
		break

capturando.release()
cv2.destroyAllWindows()