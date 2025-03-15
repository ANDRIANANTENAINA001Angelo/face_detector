from flask import Flask, request, jsonify
import cv2
import numpy as np
import requests
from facenet_pytorch import InceptionResnetV1, MTCNN, extract_face
from PIL import Image
from io import BytesIO

# Initialiser le modèle InceptionResnetV1 et MTCNN
model = InceptionResnetV1(pretrained='vggface2').eval()
mtcnn = MTCNN()

# Fonction pour extraire les embeddings faciaux
def extract_face_embeddings(face_image):
    tensor = mtcnn(face_image)
    if tensor is not None:
        embeddings = model(tensor.unsqueeze(0))
        return embeddings.detach().numpy()[0]
    return None

# Fonction pour télécharger une image à partir d'une URL
def download_image(url):
    response = requests.get(url)
    image = Image.open(BytesIO(response.content)).convert('RGB')
    image = np.array(image)
    return image

# Fonction pour comparer les embeddings faciaux
def compare_embeddings(embedding1, embedding2, threshold=1.0):
    distance = np.linalg.norm(embedding1 - embedding2)
    return distance < threshold

# Initialiser l'application Flask
app = Flask(__name__)

# Définir la route POST pour la correspondance des visages
@app.route('/correspondance', methods=['POST'])
def correspondance():
    data = request.json

    image_url_test = data['image_url']
    image_url_compare = data['profile_url']

    # Télécharger les images
    image_test = download_image(image_url_test)
    image_compare = download_image(image_url_compare)

    # Extraire les embeddings des visages
    embedding_test = extract_face_embeddings(image_test)
    embedding_compare = extract_face_embeddings(image_compare)

    if embedding_test is None or embedding_compare is None:
        return jsonify({'correspondance': 0, 'message': 'Aucun visage détecté dans une ou les deux images'})

    # Comparer les embeddings
    if compare_embeddings(embedding_test, embedding_compare):
        return jsonify({'correspondance': 1})
    else:
        return jsonify({'correspondance': 0})

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)
    # app.debug(True)

