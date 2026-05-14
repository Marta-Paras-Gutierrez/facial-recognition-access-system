# Librerías empleadas
import cv2				# Utilizar OpenCV para reconocimiento de rostros
import os				# Para crear, abrir o trabajar con directorios
import numpy as np		# Para el cálculo numérico y el análisis de los datos

# Registro de archivos sobre personas que puede hacer el entrenamiento de reconocimiento
carpetaDeTrabajo = 'G:/TFG/Programa/Rostros'
listaPersonal = os.listdir(carpetaDeTrabajo)
print('Lista de personas: ', listaPersonal)

etiquetas = []
informacionRostros = []
etiquetando = 0

# Saber cuantas personas va a reconocer y cuantas fotos hay
for nombreDirectorio in listaPersonal:
    carpetaPersonal = carpetaDeTrabajo + '/' + nombreDirectorio
    print('Leyendo las imágenes...')

    for nombreArchivo in os.listdir(carpetaPersonal):
        print('Rostros: ', nombreDirectorio + '/' + nombreArchivo)
        etiquetas.append(etiquetando)
        informacionRostros.append(cv2.imread(carpetaPersonal + '/' + nombreArchivo, 0)) # 0 para transformarlo a escala de grises
        
        # Visualizar lo que estamos haciendo
        imagen = cv2.imread(carpetaPersonal + '/' + nombreArchivo, 0)
        cv2.imshow('Visualizando', imagen)
        cv2.waitKey(10)

    etiquetando += 1

# Comprobar el número de fotos que hay para un mismo usuario/etiqueta
etiquetando = 0
for etiqueta in listaPersonal:
    print('Número de fotos de la persona con etiqueta {}: '.format(etiquetando), np.count_nonzero(np.array(etiquetas) == 0))
    etiquetando += 1
'''
# ---------------------------- Método 1 - Eigen ----------------------------
reconocimiento_facial = cv2.face.EigenFaceRecognizer_create()
print('Entrenando...')

# Entrenando el reconocimiento de rostros
    # Lista donde están almacenados todos los rostros, etiquetas de cada rostro
reconocimiento_facial.train(informacionRostros, np.array(etiquetas))

# Almacenado el modelo obtenido
   # Archivo .xml o .yaml
reconocimiento_facial.write('modeloReconocido-EigenFace.xml')
print('Modelo reconocido')
'''

# ---------------------------- Método 2 - Fisher ----------------------------
reconocimiento_facial = cv2.face.FisherFaceRecognizer_create()
print('Entrenando...')

# Entrenando el reconocimiento de rostros
    # Lista donde están almacenados todos los rostros, etiquetas de cada rostro
reconocimiento_facial.train(informacionRostros, np.array(etiquetas))

# Almacenado el modelo obtenido
    # Archivo .xml o .yaml
reconocimiento_facial.write('modeloReconocido-FisherFace.xml')
print('Modelo reconocido')

'''
# ---------------------------- Método 3 - LBPH ----------------------------
reconocimiento_facial = cv2.face.LBPHFaceRecognizer_create()
print('Entrenando...')

# Entrenando el reconocimiento de rostros
    # Lista donde están almacenados todos los rostros, etiquetas de cada rostro
reconocimiento_facial.train(informacionRostros, np.array(etiquetas))

# Almacenado el modelo obtenido
    # Archivo .xml o .yaml
reconocimiento_facial.write('modeloReconocido-LBPHFace.xml')
print('Modelo reconocido')
'''
cv2.destroyAllWindows()