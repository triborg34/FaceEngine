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
            
            # Get existing embeddings
            existing_embeddings = records['items'][0]['embdanings']
            if isinstance(existing_embeddings, str):
                existing_embeddings = json.loads(existing_embeddings)  # Parse JSON string if needed
            
            # Append the new embedding
            existing_embeddings.append(embed.tolist())  # Add the new embedding to the list
            
            print(len(existing_embeddings))  # Check how many embeddings there are now
            
            # Prepare data to update the record
            data = {
                "embdanings": json.dumps(existing_embeddings)  # Ensure it's a string when sending
            }

            # Update the record with the new embeddings
            update_url = f"http://127.0.0.1:8090/api/collections/known_face/records/{record_id}"
            update_response = requests.patch(update_url, data=data)

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
        "embdanings": json.dumps([embed_list])  # Ensure this is a list of embeddings as a string
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
                    known_faces[person].append(faces[0].embedding)





import os
import time
import cv2
import numpy as np
import threading
import queue
import requests
from ultralytics import YOLO
from insightface.app import FaceAnalysis
from sklearn.metrics.pairwise import cosine_similarity






# ---- Load YOLOv8-Face ----
yolo_face = YOLO('yolov8n-face.pt')

# ---- InsightFace (ArcFace embeddings) ----
face_embedder = FaceAnalysis(name='buffalo_l', providers=['CPUExecutionProvider'])
face_embedder.prepare(ctx_id=0)

# ---- Load known faces ----
known_faces = {}  # name -> list of embeddings


def safe_reshape(embedding, dim=512):
    # If already a nested list of [512]-dim vectors, return as-is
    if isinstance(embedding[0], list) and len(embedding[0]) == dim:
        return embedding
    
    # Otherwise, try to reshape the flat list
    if len(embedding) % dim != 0:
        raise ValueError(f"Inconsistent embedding length: {len(embedding)} not divisible by {dim}")
    
    return [embedding[i:i+dim] for i in range(0, len(embedding), dim)]

def load_embeddings_from_db():
    url = "http://127.0.0.1:8090/api/collections/known_face/records?perPage=1000"

    try:
        res = requests.get(url)
        res.raise_for_status()
        records = res.json()["items"]

        for item in records:
            name = item["name"]
            embedding = item.get("embdanings")
            print(len(embedding))
            embedding = embedding[:len(embedding) - (len(embedding) % 512)]

            if embedding:
                try:
                    reshaped = safe_reshape(embedding)
                    for emb in reshaped:
                        emb_array = np.array(emb, dtype=np.float32)
                        known_faces.setdefault(name, []).append(emb_array)
                except Exception as reshape_error:
                    print(f"⚠️ Error reshaping embedding for {name}: {reshape_error}")
        
        print(f"✅ Loaded {sum(len(v) for v in known_faces.values())} embeddings from {len(known_faces)} persons")

    except Exception as e:
        print(f"❌ Failed to load embeddings: {e}")


def recognize(embedding, threshold=0.6):
    for name, embeddings in known_faces.items():
        for known_emb in embeddings:
            sim = cosine_similarity([embedding], [known_emb])[0][0]
            if sim > threshold:
                return name, sim
    return "Unknown", 0

# Load known faces
try:
    load_embeddings_from_db()
    print("Known faces loaded.")
except Exception as e:
    print("Error loading faces:", e)

# ---- RTSP or webcam ----
cap = cv2.VideoCapture(0)  # Change to RTSP link if needed
cap.set(cv2.CAP_PROP_FPS, 30)

# ---- Recognition Thread ----
recognition_queue = queue.Queue()
face_names = {}


def recognize_thread():
    while True:
        item = recognition_queue.get()
        if item is None:
            break
        face_id, face_img = item
        faces = face_embedder.get(face_img)
        if faces:
            name, sim = recognize(faces[0].embedding)
            face_names[face_id] = f"{name} ({sim:.2f})"
        else:
            face_names[face_id] = "No face"

threading.Thread(target=recognize_thread, daemon=True).start()

# ---- Main loop ----
def mainLoop():
    frame_id = 0
    start_time = time.time()
    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        frame_id += 1
        results = yolo_face(frame)[0]

        if results.boxes is not None and len(results.boxes) > 0:
            boxes = results.boxes.data.cpu().numpy()

            for i, box in enumerate(boxes):
                x1, y1, x2, y2, conf = box[:5]
                if conf < 0.5:
                    continue

                x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])

                # Padding around face
                padding = 50
                h, w, _ = frame.shape
                x1 = max(0, x1 - padding)
                y1 = max(0, y1 - padding)
                x2 = min(w, x2 + padding)
                y2 = min(h, y2 + padding)

                face_crop = frame[y1:y2, x1:x2]

                # Only recognize every 25 frames
                if frame_id % 25 == 0:
                    recognition_queue.put((i, face_crop))

                label = face_names.get(i, "Detecting...")
               
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, label, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        fps = 1.0 / (time.time() - start_time)
        start_time = time.time()
        cv2.putText(frame, f"FPS: {fps:.2f}", (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)

        cv2.imshow("YOLO-Face Recognition", frame)
        try:
            del frame,face_crop
        except Exception as e:
            continue

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Cleanup
    recognition_queue.put(None)
    cap.release()
    cv2.destroyAllWindows()

if __name__=="__main__":
    mainLoop()