import face_recognition as fr
import os
import cv2
import numpy as np
from time import sleep

def get_encoded_faces():
    encoded = {}
    for dirpath, dnames, fnames in os.walk("./face_repository"):
        for f in fnames:
            if f.endswith(".jpg") or f.endswith(".png"):
                face = fr.load_image_file("face_repository/" + f)
                encoding = fr.face_encodings(face)[0]
                encoded[f.split(".")[0]] = encoding
    return encoded

def classify_face_from_camera():
    faces = get_encoded_faces()
    faces_encoded = list(faces.values())
    known_face_names = list(faces.keys())

    cap = cv2.VideoCapture(0)  # 0 for default camera

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error reading from the camera.")
            break

        face_locations = fr.face_locations(frame)
        unknown_face_encodings = fr.face_encodings(frame, face_locations)

        face_names = []
        for face_encoding in unknown_face_encodings:
            matches = fr.compare_faces(faces_encoded, face_encoding)
            name = "Unknown"

            face_distances = fr.face_distance(faces_encoded, face_encoding)
            best_match_index = np.argmin(face_distances)
            if matches[best_match_index]:
                name = known_face_names[best_match_index]

            face_names.append(name)

            for (top, right, bottom, left), name in zip(face_locations, face_names):
                cv2.rectangle(frame, (left-20, top-10), (right+20, bottom+15), (300, 0, 0), 2)
                cv2.rectangle(frame, (left-20, bottom-10), (right+20, bottom+15), (500, 0, 0), cv2.FILLED)
                font = cv2.FONT_HERSHEY_DUPLEX
                cv2.putText(frame, name, (left-10, bottom+10), font, 0.5, (300, 300, 300), 1)

        cv2.imshow('Whom are you looking for?', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    classify_face_from_camera()
