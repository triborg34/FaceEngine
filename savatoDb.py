import datetime
import os
import cv2
from insightface.app import FaceAnalysis
import cv2
import numpy as np
import requests
import json
from ultralytics import YOLO
from PIL import Image
import logging



logging.basicConfig(
    level=logging.DEBUG,  # Capture everything from DEBUG and above

    format='[%(asctime)s] [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler("log.txt", mode='a',
                            encoding='utf-8'),  # Append mode
        logging.StreamHandler()  # Optional: also show logs in console
    ]
)

def load_known_faces(db_folder='dbimage'):
    face_embedder = FaceAnalysis('buffalo_l', providers=[
                                 'CUDAExecutionProvider', 'CPUExecutionProvider'])
    face_embedder.prepare(ctx_id=0)
    model = YOLO('models/yolov8n.pt')
    known_faces = {}
    for person in os.listdir(db_folder):
        person_path = os.path.join(db_folder, person)
        if os.path.isdir(person_path):
            known_faces[person] = []
            for img_file in os.listdir(person_path):
                img_path = os.path.join(person_path, img_file)
                img = cv2.imread(img_path)
                frame = model(img, classes=[0])[0]
                if len(frame) > 0:

                    x1, y1, x2, y2 = map(int, frame.boxes.xyxy[0][:4])

                    img = img[y1:y2, x1:x2]

                if img is None:
                    continue
                cv2.imshow('frame', img)
                cv2.waitKey(0)
                faces = face_embedder.get(img)
                if faces:
                    embed = faces[0].embedding
                    # Check if the person already exists
                    if check_person_exists(person):
                        # TODO:SAVE CROPPED??
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
        logging.info(f" Failed to check existence of {name}: {response.status_code}")
        return False


def update_embeddings(embed, name, img_path):
    url = f"http://127.0.0.1:8090/api/collections/known_face/records?filter=name=%22{name}%22"

    response = requests.get(url)

    if response.status_code == 200:
        records = response.json()
        if len(records['items']) > 0:
            # Get the ID of the first match
            record_id = records['items'][0]['id']

            # Prepare data to update the record with the latest embedding
            data = {
                # Store only the latest embedding as a string
                "embdanings": json.dumps(embed.tolist())
            }
    #         files = {
    #     "image": open(img_path, "rb")
    # }
            with open(img_path, 'rb') as file:
                files = {"image": file}
              # Update the record with the new embedding
                update_url = f"http://127.0.0.1:8090/api/collections/known_face/records/{record_id}"
                update_response = requests.patch(
                    update_url, data=data, files=files)

                if update_response.status_code == 200:
                    logging.info(f" Updated: {name}")
                else:
                    logging.info(
                        f" Failed to update {name}: {update_response.status_code}")
                    logging.info(update_response.text)
        else:
            logging.info(f" No matching record found to update for {name}")
    else:
        logging.info(
            f" Failed to fetch record for updating {name}: {response.status_code}")


def sendToDb(embed, name, img_path):
    url = "http://127.0.0.1:8090/api/collections/known_face/records"

    # Convert embedding (numpy) to list
    embed_list = embed.tolist()

    # Prepare data and files
    data = {
        "name": name,
        "embdanings": embed_list  # Ensure this is a list of embeddings as a string
    }
    # files = {
    #     "image": open(img_path, "rb")
    # }
    with open(img_path, 'rb') as file:
        files = {"image": file}
        response = requests.post(url, data=data, files=files)

        if response.status_code == 200:
            logging.info(f"✅ Uploaded: {name}")
        else:
            logging.info(f"❌ Failed to upload {name}: {response.status_code}")
            logging.info(response.text)


def safe_reshape(embedding, dim=512):
    """
    Reshape a flat embedding list into a nested list of vectors with the specified dimension.
    """
    if isinstance(embedding[0], list) and len(embedding[0]) == dim:
        return embedding

    if len(embedding) % dim != 0:
        raise ValueError(
            f"Inconsistent embedding length: {len(embedding)} not divisible by {dim}")

    return [embedding[i:i+dim] for i in range(0, len(embedding), dim)]


def load_embeddings_from_db():
    known_names = {}
    """
    Load known face embeddings from a database and store them in the `known_names` dictionary.
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
                    logging.error(
                        f" Error reshaping embedding for {name}: {reshape_error}")

        logging.info(
            f"Loaded {sum(len(v) for v in known_names.values())} embeddings from {len(known_names)} persons")
        return known_names

    except Exception as e:
        logging.error(f" Failed to load embeddings: {e}")


tempTime = None


def savePicture(frame, croppedface, name, track_id):
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame = Image.fromarray(frame)
    frame.save(
        f'outputs/screenshot/s.{name}_{track_id}.jpg', "JPEG", quality=10, optimize=True)
    # cropp
    croppedface = cv2.cvtColor(croppedface, cv2.COLOR_BGR2RGB)
    croppedface = Image.fromarray(croppedface)
    croppedface.save(
        f'outputs/cropped/c.{name}_{track_id}.jpg', "JPEG", quality=100, optimize=True)


def timediff(current_time):
    global tempTime
    if tempTime is None:
        return True
    return (current_time - tempTime).total_seconds() >= 60


recent_names = []

def should_insert(name, track_id):
    if name == "unknown":
        return True

    now = datetime.datetime.now()
    for entry in recent_names:
        if entry['name'] == name:
            diff = now - entry['time']
            if diff.total_seconds() < 10:
                
                return False  # Found recent same person
    return True

async def insertToDb(name, frame, croppedface, score, track_id):
    global tempTime
    url = "127.0.0.1:8090/api/collections/collection/records"
    timeNow = datetime.datetime.now()
    display_time = timeNow.strftime("%H:%M:%S")
    display_date = timeNow.strftime("%Y-%m-%d")
    # Ensure the directory for saving cropped faces exists

    if not os.path.exists('outputs'):
        os.makedirs('outputs')
        os.makedirs('outputs/cropped')
        os.makedirs('outputs/screenshot')
    else:
        pass
    if should_insert(name,track_id):
        savePicture(frame,croppedface,name,track_id)
        recent_names.append({
        "name": name,
        "trackId": track_id,
        "time": datetime.datetime.now()
         })
    



if __name__ == "__main__":
    load_known_faces()
