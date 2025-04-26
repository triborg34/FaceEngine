import os
from insightface.app import FaceAnalysis
import cv2
import numpy as np
import requests
import json
from ultralytics import YOLO

face_embedder = FaceAnalysis(name='buffalo_l', root='./', providers=['CPUExecutionProvider'])
face_embedder.prepare(ctx_id=0)
model=YOLO('yolov8n-face.pt')


known_faces = {}

def load_known_faces(db_folder='dbimage'):
    for person in os.listdir(db_folder):
        person_path = os.path.join(db_folder, person)
        if os.path.isdir(person_path):
            known_faces[person] = []
            for img_file in os.listdir(person_path):
                img_path = os.path.join(person_path, img_file)
                img = cv2.imread(img_path)
                frame=model(img)[0]
                if len(frame)>0:
                    
                    
                    x1, y1, x2, y2 = map(int, frame.boxes.xyxy[0][:4])
                    padding = 100
                    h, w, _ = img.shape
                    x1 = max(0, x1 - padding)
                    y1 = max(0, y1 - padding)
                    x2 = min(w, x2 + padding)
                    y2 = min(h, y2 + padding)

                    img = img[y1:y2, x1:x2]
                    cv2.imshow("face", img)
                    cv2.waitKey(0)
                        
                if img is None:
                    continue
                faces = face_embedder.get(img)
                print(faces)
                if faces:
                    embed = faces[0].embedding
                    # Check if the person already exists
                    if check_person_exists(person):
                        update_embeddings(embed, person, img_path)
                    else:
                        sendToDb(embed, person, img_path)

def check_person_exists(name):
    url = f"http://127.0.0.1:8090/api/collections/known_face/records?filter=name=%22{name}%22"
    
    response = requests.get(url)

    if response.status_code == 200:
        records = response.json()
        return len(records['items']) > 0  # If we found any matching record
    else:
        print(f"❌ Failed to check existence of {name}: {response.status_code}")
        return False

def update_embeddings(embed, name, img_path):
    url = f"http://127.0.0.1:8090/api/collections/known_face/records?filter=name=%22{name}%22"
    print(url)
    response = requests.get(url)

    if response.status_code == 200:
        records = response.json()
        if len(records['items']) > 0:
            record_id = records['items'][0]['id']  # Get the ID of the first match
            
            # Prepare data to update the record with the latest embedding
            data = {
                "embdanings": json.dumps(embed.tolist())  # Store only the latest embedding as a string
            }
            files = {
        "image": open(img_path, "rb")
    }

            # Update the record with the new embedding
            update_url = f"http://127.0.0.1:8090/api/collections/known_face/records/{record_id}"
            update_response = requests.patch(update_url, data=data,files=files)

            if update_response.status_code == 200:
                print(f"✅ Updated: {name}")
            else:
                print(f"❌ Failed to update {name}: {update_response.status_code}")
                print(update_response.text)
        else:
            print(f"❌ No matching record found to update for {name}")
    else:
        print(f"❌ Failed to fetch record for updating {name}: {response.status_code}")

def sendToDb(embed, name, img_path):
    url = "http://127.0.0.1:8090/api/collections/known_face/records"

    # Convert embedding (numpy) to list
    embed_list = embed.tolist()
    
    # Prepare data and files
    data = {
        "name": name,
        "embdanings":embed_list # Ensure this is a list of embeddings as a string
    }
    files = {
        "image": open(img_path, "rb")
    }

    response = requests.post(url, data=data, files=files)

    if response.status_code == 200:
        print(f"✅ Uploaded: {name}")
    else:
        print(f"❌ Failed to upload {name}: {response.status_code}")
        print(response.text)


def safe_reshape(embedding, dim=512):
    """
    Reshape a flat embedding list into a nested list of vectors with the specified dimension.
    """
    if isinstance(embedding[0], list) and len(embedding[0]) == dim:
        return embedding
    
    if len(embedding) % dim != 0:
        raise ValueError(f"Inconsistent embedding length: {len(embedding)} not divisible by {dim}")
    
    return [embedding[i:i+dim] for i in range(0, len(embedding), dim)]



def load_embeddings_from_db():
    known_names = {}
    """
    Load known face embeddings from a database and store them in the `known_faces` dictionary.
    """
    url = "http://127.0.0.1:8090/api/collections/known_face/records?perPage=1000"

    try:
        res = requests.get(url)
        res.raise_for_status()
        records = res.json()["items"]

        for item in records:
            name = item["name"]
            embedding = item.get("embdanings")
            if embedding:
                embedding = embedding[:len(embedding) - (len(embedding) % 512)]
                try:
                    reshaped = safe_reshape(embedding)
                    for emb in reshaped:
                        emb_array = np.array(emb, dtype=np.float32)
                        known_names.setdefault(name, []).append(emb_array)
                except Exception as reshape_error:
                    print(f"⚠️ Error reshaping embedding for {name}: {reshape_error}")
        
        print(f"✅ Loaded {sum(len(v) for v in known_names.values())} embeddings from {len(known_names)} persons")
        return known_names

    except Exception as e:
        print(f"❌ Failed to load embeddings: {e}")

if __name__ == "__main__":
    load_known_faces()
