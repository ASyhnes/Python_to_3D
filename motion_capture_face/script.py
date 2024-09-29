import cv2
import mediapipe as mp
import os
import numpy as np

mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils

input_video_path = 'test_visage.mp4'
cap = cv2.VideoCapture(input_video_path)
if not cap.isOpened():
    print("Erreur : Impossible d'ouvrir la vidéo.")
    exit()

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

base_name = os.path.splitext(os.path.basename(input_video_path))[0]
output_video_path = f"{base_name}_points.avi"

fourcc = cv2.VideoWriter_fourcc(*'XVID') 
out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

with mp_face_mesh.FaceMesh(static_image_mode=False,
                            max_num_faces=1,
                            refine_landmarks=True) as face_mesh:
    
    while True:
        success, frame = cap.read()
        if not success:
            print("Fin de la vidéo ou problème de lecture.")
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        results = face_mesh.process(frame_rgb)

        output_frame = np.zeros((height, width, 3), dtype=np.uint8)

        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                for landmark in face_landmarks.landmark:
                    x = int(landmark.x * width)
                    y = int(landmark.y * height)
                    cv2.circle(output_frame, (x, y), 3, (0, 255, 0), -1)  

        out.write(output_frame)

        cv2.imshow('Points de repère', output_frame)


        if cv2.waitKey(int(1000 / fps)) & 0xFF == ord('q'):
            break

cap.release()
out.release()  
cv2.destroyAllWindows()

print(f"Vidéo des points enregistrée sous : {output_video_path}")
