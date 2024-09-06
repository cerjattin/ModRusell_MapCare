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
    r = 5
    t = 3

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
                    # ceja derecha
                    x1, y1 = lista[65][1:]
                    x2, y2 = lista[158][1:]
                    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                    longitud1 = math.hypot(x2 - x1, y2 - y1)

                    # ceja izquierda
                    x3, y3 = lista[295][1:]
                    x4, y4 = lista[385][1:]
                    cx2, cy2 = (x3 + x4) // 2, (y3 + y4) // 2
                    longitud2 = math.hypot(x4 - x3, y4 - y3)

                    # boca extremos
                    x5, y5 = lista[78][1:]
                    x6, y6 = lista[308][1:]
                    cx3, cy3 = (x5 + x6) // 2, (y5 + y6) // 2
                    longitud3 = math.hypot(x6 - x5, y6 - y5)

                    # boca apertura
                    x7, y7 = lista[13][1:]
                    x8, y8 = lista[14][1:]
                    cx4, cy4 = (x7 + x8) // 2, (y7 + y8) // 2
                    longitud4 = math.hypot(x8 - x7, y8 - y7)

                    # Variable para capturar las coordenadas
                    emocion_detectada = False
                    coordenadas_emocion = {}

                    # Clasificación de emociones
                    # Molesto
                    if longitud1 < 19 and longitud2 < 19 and longitud3 > 80 and longitud3 < 95 and longitud4 < 5:
                        cv2.putText(frame, 'Persona molesta', (480, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
                        emocion_detectada = True
                        coordenadas_emocion = {'ceja_derecha': (x1, y1), 'ceja_izquierda': (x3, y3), 'boca_extremos': (x5, y5)}

                    # Feliz
                    elif longitud1 > 20 and longitud1 < 30 and longitud2 > 20 and longitud2 < 30 and longitud3 < 30 and longitud3 > 109 and longitud4 > 10 and longitud4 < 20:
                        cv2.putText(frame, 'Persona feliz', (480, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
                        emocion_detectada = True
                        coordenadas_emocion = {'ceja_derecha': (x1, y1), 'ceja_izquierda': (x3, y3), 'boca_extremos': (x5, y5)}

                    # Asombrado
                    elif longitud1 > 35 and longitud2 > 35 and longitud3 > 85 and longitud3 < 90 and longitud4 > 20:
                        cv2.putText(frame, 'Persona asombrada', (480, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
                        emocion_detectada = True
                        coordenadas_emocion = {'ceja_derecha': (x1, y1), 'ceja_izquierda': (x3, y3), 'boca_extremos': (x5, y5)}

                    # Triste
                    elif longitud1 > 25 and longitud1 < 35 and longitud2 > 25 and longitud2 < 35 and longitud3 > 90 and longitud3 < 95 and longitud4 < 5:
                        cv2.putText(frame, 'Persona triste', (480, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
                        emocion_detectada = True
                        coordenadas_emocion = {'ceja_derecha': (x1, y1), 'ceja_izquierda': (x3, y3), 'boca_extremos': (x5, y5)}

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
