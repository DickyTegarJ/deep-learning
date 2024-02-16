import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Fungsi untuk mendeteksi wajah dan menerapkan model deteksi masker
def detect_mask(frame, model):
    # Lakukan deteksi wajah menggunakan Haar Cascade Classifier
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.3, minNeighbors=5)

    for (x, y, w, h) in faces:
        face_roi = frame[y:y + h, x:x + w]
        face_roi = cv2.resize(face_roi, (64, 64))
        face_roi = face_roi / 255.0
        face_roi = face_roi.reshape(1, 64, 64, 3)

        # Prediksi menggunakan model
        result = model.predict(face_roi)
        label = np.argmax(result)

        # Tampilkan hasil prediksi
        if label == 0:
            cv2.putText(frame, "Without Mask", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
        else:
            cv2.putText(frame, "With Mask", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    return frame

# Muat model
model = load_model('face_mask_detection_model.h5')  # Pastikan model sudah dilatih sebelumnya

# Buka kamera laptop
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()

    if not ret:
        break

    # Deteksi masker pada setiap frame
    frame_with_mask = detect_mask(frame, model)

    # Tampilkan frame
    cv2.imshow('Face Mask Detection', frame_with_mask)

    # Hentikan program jika tombol 'q' ditekan
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Tutup kamera dan jendela
cap.release()
cv2.destroyAllWindows()
