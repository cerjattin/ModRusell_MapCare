import cv2
import mediapipe as mp
import math
import time

cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)

mpDibujo = mp.solutions.drawing_utils
ConfDibu = mpDibujo.DrawingSpec(thickness=1, circle_radius=1)

mpMallaFacial = mp.solutions.face_mesh
MallaFacial = mpMallaFacial.FaceMesh(max_num_faces=1)

while True:
    ret, frame = cap.read()
    frameRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    resultados = MallaFacial.process(frameRGB)

    px = []
    py = []
    lista = []

    if resultados.multi_face_landmarks:
        for rostros in resultados.multi_face_landmarks:
            mpDibujo.draw_landmarks(frame, rostros, mpMallaFacial.FACEMESH_TESSELATION, ConfDibu, ConfDibu)

            for id, puntos in enumerate(rostros.landmark):
                al, an, c = frame.shape
                x, y = int(puntos.x * an), int(puntos.y * al)
                px.append(x)
                py.append(y)
                lista.append([id, x, y])

                if len(lista) == 468:

                    # Frente
                    x9, y9 = lista[10][1:]
                    x10, y10 = lista[151][1:]
                    x12, y12 = lista[100][1:]
                    x13, y13 = lista[336][1:]

                    #10 to left
                    #x10_izq ,y10_izq= lista[5][1:] punto de la nariz
                    x10_lef ,y10_lef= lista[104][1:] 
                    x10_right ,y10_right= lista[333][1:]

                    # Calcular distancias (longitud de la frente)
                    longitud_frente_horizontal = math.hypot(x10 - x9, y10 - y9)  # Distancia horizontal de la frente
                   # longitud_frente_vertical = math.hypot(x12 - x11, y12 - y11)  # Distancia vertical de la frente

                    cv2.circle(frame, (x9, y9), 5, (255, 0, 0), -1)  # Punto 9
                    cv2.circle(frame, (x10, y10), 5, (255, 0, 0), -1)  # Punto 10
                   
                    cv2.circle(frame, (x9, y9), 5, (255, 0, 0), -1)    # Punto 9
                    cv2.circle(frame, (x10_lef, y10_lef), 5, (255, 0, 0), -1) 
                    cv2.circle(frame, (x10_right, y10_right), 5, (255, 0, 0), -1) 
                    # Dibuja líneas para la frente
                    cv2.line(frame, (x9, y9), (x10, y10), (0, 255, 0), 2)  
                    cv2.line(frame, (x10_lef, y10_lef), (x10, y10), (0, 255, 0), 2)
                    cv2.line(frame, (x10_right, y10_right), (x10, y10), (0, 255, 0), 2)
                  
                    # Cejas
                    x1, y1 = lista[65][1:]
                    x2, y2 = lista[158][1:]
                    longitud1 = math.hypot(x2 - x1, y2 - y1)

                    x3, y3 = lista[295][1:]
                    x4, y4 = lista[385][1:]
                    longitud2 = math.hypot(x4 - x3, y4 - y3)

                    # Boca
                    x5, y5 = lista[78][1:]
                    x6, y6 = lista[308][1:]
                    longitud3 = math.hypot(x6 - x5, y6 - y5)
                    
                    x7, y7 = lista[13][1:]
                    x8, y8 = lista[14][1:]
                    longitud4 = math.hypot(x8 - x7, y8 - y7)
                    
                    # Variable para capturar las coordenadas
                    emocion_detectada = False
                    coordenadas_emocion = {}


                 #   for id, puntos in enumerate(rostros.landmark):
                  #   al, an, c = frame.shape
                   #  x, y = int(puntos.x * an), int(puntos.y * al)
                    # cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)  # Dibuja todos los puntos en verde
                     #cv2.putText(frame, f'{id}', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

                    # Clasificación de emociones
                    # Molesto
                    if longitud1 < 19 and longitud2 < 19 and longitud3 > 80 and longitud3 < 95 and longitud4 < 5 :
                        cv2.putText(frame, 'Persona molesta', (480, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
                        emocion_detectada = True
                        coordenadas_emocion = {'ceja_derecha': (x1, y1), 'ceja_izquierda': (x3, y3), 'boca_extremos': (x5, y5), 'frente': (x9, y9)}

                    # Feliz
                    elif longitud1 > 20 and longitud1 < 30 and longitud2 > 20 and longitud2 < 30 and longitud3 < 30 and longitud3 > 109 and longitud4 > 10 and longitud4 < 20  :
                        cv2.putText(frame, 'Persona feliz', (480, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
                        emocion_detectada = True
                        coordenadas_emocion = {'ceja_derecha': (x1, y1), 'ceja_izquierda': (x3, y3), 'boca_extremos': (x5, y5), 'frente': (x9, y9)}

                    # Asombrado
                    elif longitud1 > 35 and longitud2 > 35 and longitud3 > 85 and longitud3 < 90 and longitud4 > 20  :
                        cv2.putText(frame, 'Persona asombrada', (480, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
                        emocion_detectada = True
                        coordenadas_emocion = {'ceja_derecha': (x1, y1), 'ceja_izquierda': (x3, y3), 'boca_extremos': (x5, y5), 'frente': (x9, y9)}

                    # Triste
                    elif longitud1 > 25 and longitud1 < 35 and longitud2 > 25 and longitud2 < 35 and longitud3 > 90 and longitud3 < 95 and longitud4 < 5  :
                        cv2.putText(frame, 'Persona triste', (480, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
                        emocion_detectada = True
                        coordenadas_emocion = {'ceja_derecha': (x1, y1), 'ceja_izquierda': (x3, y3), 'boca_extremos': (x5, y5), 'frente': (x9, y9)}

                    if emocion_detectada:
                        # Mostrar coordenadas en consola
                        print(f"Coordenadas de la emoción detectada: {coordenadas_emocion}")
                        
                        # Mostrar las coordenadas en la pantalla
                        for parte, (x, y) in coordenadas_emocion.items():
                            cv2.putText(frame, f"{parte}: ({x}, {y})", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

                        # Pausa de 5 segundos
                        cv2.imshow("Analisis de sentimientos con malla facial", frame)
                        time.sleep(5)

                        

    cv2.imshow("Analisis de sentimientos con malla facial", frame)
    t = cv2.waitKey(1)

    if t == 27:  # Presionar 'Esc' para salir
        break

cap.release()
cv2.destroyAllWindows()
