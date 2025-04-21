import os
from insightface.app import FaceAnalysis
import cv2
import requests
import json

face_embedder = FaceAnalysis(name='buffalo_l', providers=['CPUExecutionProvider'])
face_embedder.prepare(ctx_id=0)

known_faces = {}

def load_known_faces(db_folder='dbimage'):
    for person in os.listdir(db_folder):
        person_path = os.path.join(db_folder, person)
        if os.path.isdir(person_path):
            known_faces[person] = []
            for img_file in os.listdir(person_path):
                img_path = os.path.join(person_path, img_file)
                img = cv2.imread(img_path)
                if img is None:
                    continue
                faces = face_embedder.get(img)
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

load_known_faces()
