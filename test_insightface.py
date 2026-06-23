from insightface.app import FaceAnalysis
import cv2

# Inicializar modelo usando solo CPU
app = FaceAnalysis(providers=['CPUExecutionProvider'])
app.prepare(ctx_id=-1, det_size=(640, 640))

img = cv2.imread("foto.jpg")
if img is None:
    raise FileNotFoundError("No se encontró la imagen 'foto.jpg' en el directorio actual.")

faces = app.get(img)

print(f"Rostros encontrados: {len(faces)}")

for face in faces:
    print(face.bbox)