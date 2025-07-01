import datetime
import os
from typing import NamedTuple
from urllib.request import urlopen
import cv2
from insightface.app import FaceAnalysis
import cv2
import numpy as np
import requests
import json
from ultralytics import YOLO
from PIL import Image
import logging

import urllib.request


logging.basicConfig(
    level=logging.DEBUG,  # Capture everything from DEBUG and above

    format='[%(asctime)s] [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler("log.txt", mode='a',
                            encoding='utf-8'),  # Append mode
        logging.StreamHandler()  # Optional: also show logs in console
    ]
)


def reciveFromUi(name, imagePath, age, gender, role, socialnumber, isUrl):
    """
    Receive data from the UI and process it.
    """
    face_embedder = FaceAnalysis('buffalo_l', providers=[
        'CUDAExecutionProvider', 'CPUExecutionProvider'])
    face_embedder.prepare(ctx_id=0)
    model = YOLO('models/yolov8n.pt')
    if isUrl:
        path = urllib.request.urlretrieve(
            imagePath, "uploads/local-filename.jpg")
        imagePath = path[0]

    img = cv2.imread(imagePath)
    frame = model(img, classes=[0])[0]
    if len(frame.boxes) > 0:
        x1, y1, x2, y2 = map(int, frame.boxes.xyxy[0][:4])
        img = img[y1:y2, x1:x2]
    if img is None:
        logging.error(f"Image not found at {imagePath}")
        return
    face = face_embedder.get(img)
    if face:
        embed = face[0].embedding
        print(embed)

        # Check if the person already exists
        if check_person_exists(name):
            update_embeddings(embed, name, imagePath, age,
                              gender, role, socialnumber)
        else:
            sendToDb(embed, name, imagePath, age, gender,
                     role, socialnumber)


def check_person_exists(name):
    url = f"http://127.0.0.1:8090/api/collections/known_face/records?filter=name=%22{name}%22"

    response = requests.get(url)

    if response.status_code == 200:
        records = response.json()
        return len(records['items']) > 0  # If we found any matching record
    else:
        logging.info(
            f" Failed to check existence of {name}: {response.status_code}")
        return False


def update_embeddings(embed, name, img_path, age, gender, role, socialnumber):
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
                "embdanings": json.dumps(embed.tolist()), "name": name,
                "gender": gender,
                "age": age,
                "role": role,
                "socialnumber": socialnumber
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
            os.remove(img_path)
        else:
            logging.info(f" No matching record found to update for {name}")
    else:
        logging.info(
            f" Failed to fetch record for updating {name}: {response.status_code}")


def sendToDb(embed, name, img_path, age, gender, role, socialnumber):
    url = "http://127.0.0.1:8090/api/collections/known_face/records"

    # Convert embedding (numpy) to list
    embed_list = embed.tolist()

    # Prepare data and files
    data = {
        "name": name,
        "embdanings": embed_list  # Ensure this is a list of embeddings as a string
        , "gender": gender,
        "age": age,
        "role": role,
        "socialnumber": socialnumber

    }
    # files = {
    #     "image": open(img_path, "rb")
    # }
    with open(img_path, 'rb') as file:
        files = {"image": file}
        response = requests.post(url, data=data, files=files)

        if response.status_code == 200:
            logging.info(f" Uploaded: {name}")

        else:
            logging.info(f" Failed to upload {name}: {response.status_code}")
            logging.info(response.text)
    os.remove(img_path)


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
    Each entry contains name, age, gender, and embeddings.
    """
    url = "http://127.0.0.1:8090/api/collections/known_face/records?perPage=1000"

    try:
        res = requests.get(url)
        res.raise_for_status()
        records = res.json()["items"]

        for item in records:
            name = item["name"]
            # Note: typo in original - should be "embeddings"
            embedding = item.get("embdanings")
            age = item.get('age')
            gender = item.get('gender')
            role = item.get('role')

            print(f"{age=},{gender=}")

            if embedding:
                embedding = embedding[:len(embedding) - (len(embedding) % 512)]
                try:
                    reshaped = safe_reshape(embedding)

                    # Initialize the person's entry if it doesn't exist
                    if name not in known_names:
                        known_names[name] = {

                            'age': age,
                            'gender': gender,
                            'role': role,
                            'embeddings': []
                        }

                    # Add all embeddings for this person
                    for emb in reshaped:
                        emb_array = np.array(emb, dtype=np.float32)
                        known_names[name]['embeddings'].append(emb_array)

                except Exception as reshape_error:
                    logging.error(
                        f"Error reshaping embedding for {name}: {reshape_error}")

        total_embeddings = sum(len(person['embeddings'])
                               for person in known_names.values())
        logging.info(
            f"Loaded {total_embeddings} embeddings from {len(known_names)} persons")
        return known_names

    except Exception as e:
        logging.error(f"Failed to load embeddings: {e}")
        return {}


tempTime = None


def savePicture(frame, croppedface, humancrop, name, track_id):
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame = Image.fromarray(frame)
    frame_loc = f'outputs/screenshot/s.{name}_{track_id}.jpg'
    frame.save(
        f'{frame_loc}', "JPEG", quality=100, optimize=True)
    # cropp
    croppedface = cv2.cvtColor(croppedface, cv2.COLOR_BGR2RGB)
    croppedface = Image.fromarray(croppedface)
    crop_loc = f'outputs/cropped/c.{name}_{track_id}.jpg'
    croppedface.save(
        f'{crop_loc}', "JPEG", quality=100, optimize=True)
    humancrop = cv2.cvtColor(croppedface, cv2.COLOR_BGR2RGB)
    humancrop = Image.fromarray(croppedface)
    human_loc = f'outputs/cropped/c.{name}_{track_id}.jpg'
    humancrop.save(
        f'{human_loc}', "JPEG", quality=100, optimize=True)

    return frame_loc, crop_loc, human_loc


def timediff(current_time):
    global tempTime
    if tempTime is None:
        return True
    return (current_time - tempTime).total_seconds() >= 60


class RecentEntry(NamedTuple):
    name: str
    track_id: int
    time: datetime.datetime


recent_names: list[RecentEntry] = []
TIME_THRESHOLD = 10


def clean_old_entries():
    now = datetime.datetime.now()
    recent_names[:] = [
        entry for entry in recent_names
        if (now - entry.time).total_seconds() < TIME_THRESHOLD
    ]


def should_insert(name, track_id):
    now = datetime.datetime.now()
    clean_old_entries()

    for entry in recent_names:
        if name == "unknown" and entry.name == "unknown":
            if entry.track_id == track_id:
                if (now - entry.time).total_seconds() < TIME_THRESHOLD:
                    return False

        elif entry.name == name:
            if (now - entry.time).total_seconds() < TIME_THRESHOLD:
                return False

    return True


async def insertToDb(name, frame, croppedface, humancrop, score, track_id, gender, age, role, path):
    global tempTime
    url = "http://127.0.0.1:8090/api/collections/collection/records"
    timeNow = datetime.datetime.now()
    display_time = timeNow.strftime("%H:%M:%S")
    display_date = timeNow.strftime("%Y-%m-%d")
    # Ensure the directory for saving cropped faces exists

    if not os.path.exists('outputs'):
        os.makedirs('outputs')
        os.makedirs('outputs/cropped')
        os.makedirs('outputs/screenshot')
        os.makedirs('outputs/humancrop')
    else:
        pass
    if should_insert(name, track_id):
        frame_loc, crop_loc, human_loc = savePicture(
            frame, croppedface, humancrop, name, track_id)

        recent_names.append(RecentEntry(
            name=name, track_id=track_id, time=datetime.datetime.now()))

        with open(frame_loc, "rb") as file1, open(crop_loc, "rb") as file2,open(human_loc,'rb') as file3:
            files = {
                # Change field name if needed
                "frame": (frame_loc, file1, "image/jpeg"),
                # Change field name if needed
                "cropped_frame": (crop_loc, file2, "image/jpeg"),
                "humancrop":(human_loc,file3,"image/jpeg")
            }

            response = requests.post(url, files=files, data={
                "name": name,
                "score": score,
                'gender': gender,
                'age': age,
                'camera': path,
                'date': display_date,
                'time': display_time,
                'role': role,
                "track_id": str(track_id)
            })
        if response.status_code in [200, 201]:

            logging.info(response.json()['id'])
        else:
            logging.error("Error:", response.text)


def log_detection(name, track_id):
    if should_insert(name, track_id):
        recent_names.append({
            'name': name,
            'track_id': track_id,
            'time': datetime.datetime.now()
        })
        print(f" Inserted: {name} (track_id: {track_id})")
    else:
        print(f" Skipped: {name} (track_id: {track_id})")


if __name__ == "__main__":
    pass
    # import time
    # log_detection("unknown", 1)  # Should insert
    # time.sleep(2)
    # log_detection("unknown", 1)  # Should skip (too soon)
    # time.sleep(9)
    # log_detection("unknown", 1)  # Should insert (past 10 sec)
    # load_known_faces()


# def load_known_faces(db_folder='dbimage'):
#     face_embedder = FaceAnalysis('buffalo_l', providers=[
#                                  'CUDAExecutionProvider', 'CPUExecutionProvider'])
#     face_embedder.prepare(ctx_id=0)
#     model = YOLO('models/yolov8n.pt')
#     known_faces = {}
#     for person in os.listdir(db_folder):
#         person_path = os.path.join(db_folder, person)
#         if os.path.isdir(person_path):
#             known_faces[person] = []
#             for img_file in os.listdir(person_path):
#                 img_path = os.path.join(person_path, img_file)
#                 img = cv2.imread(img_path)
#                 frame = model.predict(img, classes=[0])[0]
#                 if len(frame) > 0:

#                     x1, y1, x2, y2 = map(int, frame.boxes.xyxy[0][:4])

#                     img = img[y1:y2, x1:x2]

#                 if img is None:
#                     continue
#                 cv2.imshow('frame', img)
#                 cv2.waitKey(0)
#                 faces = face_embedder.get(img)
#                 if faces:
#                     embed = faces[0].embedding
#                     # Check if the person already exists
#                     if check_person_exists(person):
#                         # TODO:SAVE CROPPED??
#                         update_embeddings(embed, person, img_path)
#                     else:
#                         sendToDb(embed, person, img_path)
