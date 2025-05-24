import cv2

# Inicia la cámara
cap = cv2.VideoCapture(0)

# Captura el primer frame como referencia
ret, frame1 = cap.read()
frame1_gray = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
frame1_gray = cv2.GaussianBlur(frame1_gray, (21, 21), 0)

while True:
    # Captura el siguiente frame
    ret, frame2 = cap.read()
    if not ret:
        break

    # Preprocesamiento: convertir a escala de grises y aplicar desenfoque
    gray = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (21, 21), 0)

    # Comparar el frame actual con el de referencia
    diff = cv2.absdiff(frame1_gray, gray)
    thresh = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)[1]
    thresh = cv2.dilate(thresh, None, iterations=2)

    # Encontrar contornos (zonas de cambio/movimiento)
    contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        if cv2.contourArea(contour) < 500:
            continue
        (x, y, w, h) = cv2.boundingRect(contour)
        cv2.rectangle(frame2, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Mostrar el resultado
    cv2.imshow("Movimiento", frame2)

    # Salir si se presiona ESC
    key = cv2.waitKey(30)
    if key == 27:
        break

# Liberar recursos
cap.release()
cv2.destroyAllWindows()
