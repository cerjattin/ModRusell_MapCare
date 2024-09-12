import cv2
import mediapipe as mp
import math
import time

# Inicializa la cámara
cap = cv2.VideoCapture(0)

# Obtener el tamaño original de la cámara
ancho = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
alto = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

print(f"Resolución de la cámara: {ancho}x{alto}")

# Configuraciones de dibujo y malla facial
mpDibujo = mp.solutions.drawing_utils
ConfDibu = mpDibujo.DrawingSpec(thickness=1, circle_radius=1)
mpMallaFacial = mp.solutions.face_mesh
MallaFacial = mpMallaFacial.FaceMesh(max_num_faces=1)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    frameRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    resultados = MallaFacial.process(frameRGB)

    lista = []

    if resultados.multi_face_landmarks:
        for rostros in resultados.multi_face_landmarks:
            mpDibujo.draw_landmarks(frame, rostros, mpMallaFacial.FACEMESH_TESSELATION, ConfDibu, ConfDibu)

            for id, puntos in enumerate(rostros.landmark):
                x, y = int(puntos.x * ancho), int(puntos.y * alto)
                lista.append([id, x, y])

                if len(lista) == 468:
                    # inicia lógica de detección de emociones
                    # Calcular las longitudes
                    def calcular_distancia(p1, p2):
                        return math.hypot(p2[0] - p1[0], p2[1] - p1[1])
                    
                    #####################################################
                    
                    
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
                    cv2.line(frame, (int(x9), int(y9)), (int(x10), int(y10)), (0, 255, 0), 2)
                    cv2.line(frame, (int(x10_lef), int(y10_lef)), (int(x10), int(y10)), (0, 255, 0), 2)
                    cv2.line(frame, (int(x10_right), int(y10_right)), (int(x10), int(y10)), (0, 255, 0), 2)

                    
                    
                    
                    ####################################################
                    # Coordenadas clave
                    ceja_derecha = (lista[65][1:], lista[158][1:])
                    pomulo_derecho = (lista[93][1:],lista[132][1:])
                    ceja_izquierda = (lista[295][1:], lista[385][1:])
                    pomulo_izq = (lista[305][1:],lista[420][1:])
                    boca_extremos = (lista[78][1:], lista[308][1:])
                    boca_apertura = (lista[13][1:], lista[14][1:])

                    
                    # punto pomulo derecho
                    x1, y1 = lista[50][1:]  # Coordenadas del punto 50
                    #pruebas puntos al rededor pomulo
                    x2, y2 = lista[123][1:]
                    cv2.circle(frame, (x2, y2), 5, (0, 255, 0), -1)
                    x4,y4= lista [207][1:]
                    cv2.circle(frame, (x4, y4), 5, (0, 255, 0), -1)
                    
                    
                    #punto pomulo izq
                    x3, y3 = lista[280][1:]  # Coordenadas del punto 280 
                    

                    # Dibujar círculo en el punto del pómulo derecho
                    cv2.circle(frame, (x1, y1), 5, (0, 255, 0), -1)  # Círculo en el punto 50
                    
                    # Dibujar círculo en el punto del pómulo izq
                    cv2.circle(frame, (x3, y3), 5, (0, 255, 0), -1)  # Círculo en el punto 280
                    

                    # Longitudes
                    longitud_ceja_derecha = calcular_distancia(*ceja_derecha)                    
                    longitud_pomder = calcular_distancia(*pomulo_derecho)
                    longitud_ceja_izquierda = calcular_distancia(*ceja_izquierda)
                    longitud_pomizq = calcular_distancia(*pomulo_izq)
                    longitud_boca_extremos = calcular_distancia(*boca_extremos)
                    longitud_boca_apertura = calcular_distancia(*boca_apertura)

                    # Clasificación de emociones
                    emocion = None
                    coordenadas_emocion = {}

                    if (longitud_ceja_derecha < 19 and longitud_ceja_izquierda < 19 and 
                        80 < longitud_boca_extremos < 95 and longitud_boca_apertura < 5):
                        emocion = 'Persona molesta'
                        coordenadas_emocion = {'ceja_derecha': ceja_derecha[0], 'ceja_izquierda': ceja_izquierda[0], 'boca_extremos': boca_extremos[0]}
                    elif (20 < longitud_ceja_derecha < 30 and 20 < longitud_ceja_izquierda < 30 and 
                          109 < longitud_boca_extremos < 30 and 10 < longitud_boca_apertura < 20):
                        emocion = 'Persona feliz'
                        coordenadas_emocion = {'ceja_derecha': ceja_derecha[0], 'ceja_izquierda': ceja_izquierda[0], 'boca_extremos': boca_extremos[0]}
                    elif (longitud_ceja_derecha > 35 and longitud_ceja_izquierda > 35 and 
                          85 < longitud_boca_extremos < 90 and longitud_boca_apertura > 20):
                        emocion = 'Persona asombrada'
                        coordenadas_emocion = {'ceja_derecha': ceja_derecha[0], 'ceja_izquierda': ceja_izquierda[0], 'boca_extremos': boca_extremos[0]}
                    elif (25 < longitud_ceja_derecha < 35 and 25 < longitud_ceja_izquierda < 35 and 
                          90 < longitud_boca_extremos < 95 and longitud_boca_apertura < 5):
                        emocion = 'Persona triste'
                        coordenadas_emocion = {'ceja_derecha': ceja_derecha[0], 'ceja_izquierda': ceja_izquierda[0], 'boca_extremos': boca_extremos[0]}
                        
                        
                    
                    if emocion:
                        # Mostrar la emoción detectada
                        cv2.putText(frame, emocion, (480, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
                        print(f"Coordenadas de la emoción detectada: {coordenadas_emocion}")
                        
                        for parte, (x, y) in coordenadas_emocion.items():
                            cv2.putText(frame, f"{parte}: ({x}, {y})", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    # Mostrar el frame
    cv2.imshow("Analisis de sentimientos con malla facial", frame)

    if cv2.waitKey(1) & 0xFF == 27:  # Presiona 'Esc' para salir
        break

cap.release()
cv2.destroyAllWindows()
