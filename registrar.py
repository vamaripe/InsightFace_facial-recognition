from insightface.app import FaceAnalysis
import cv2
import numpy as np

app = FaceAnalysis()
app.prepare(ctx_id=0)

img = cv2.imread("aprendiz.jpg")

faces = app.get(img)

if len(faces) == 0:
    print("No se encontró rostro")
    exit()

embedding = faces[0].embedding

np.save("usuario_1001.npy", embedding)

print("Usuario registrado")