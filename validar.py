from insightface.app import FaceAnalysis
import cv2
import numpy as np

app = FaceAnalysis()
app.prepare(ctx_id=0)

# Rostro registrado
emb_registrado = np.load("usuario_1001.npy")

# Nueva foto
img = cv2.imread("prueba.jpg")

faces = app.get(img)

if len(faces) == 0:
    print("No se detectó rostro")
    exit()

emb_actual = faces[0].embedding

# Similitud coseno
sim = np.dot(
    emb_registrado,
    emb_actual
) / (
    np.linalg.norm(emb_registrado)
    * np.linalg.norm(emb_actual)
)

print("Similitud:", sim)

if sim > 0.6:
    print("ACCESO PERMITIDO")
else:
    print("ACCESO DENEGADO")