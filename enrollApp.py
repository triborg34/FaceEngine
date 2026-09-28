# -*- coding: utf-8 -*-
"""
برنامه ثبت چهره (Face Enrollment App)
--------------------------------------
- اتصال به دوربین RTSP
- گرفتن عکس
- تشخیص افراد با YOLO
- انتخاب فرد مورد نظر با کلیک
- استخراج امبدینگ چهره با InsightFace
- ارسال به PocketBase (کالکشن known_face)

نصب پیش‌نیازها:
    pip install opencv-python pillow requests numpy ultralytics insightface onnxruntime

نکته: اگر برای این کالکشن قبلاً رکوردی با همین نام وجود داشته باشد،
امبدینگ جدید به لیست امبدینگ‌های همان فرد اضافه می‌شود (append)،
نه اینکه رکورد تکراری ساخته شود.
"""

import io
import json
import threading
import time

import cv2
import numpy as np
import requests
import tkinter as tk
from tkinter import ttk, messagebox
from PIL import Image, ImageTk

from ultralytics import YOLO
from insightface.app import FaceAnalysis

# ---------------------------------------------------------------------------
# تنظیمات قابل تغییر
# ---------------------------------------------------------------------------
POCKETBASE_URL = "http://127.0.0.1:8091"
COLLECTION = "known_face"
YOLO_MODEL_PATH = "models/yolov8n.pt"
FACE_MODEL_NAME = "buffalo_l"   # برای استفاده تجاری از antelopev2 استفاده نکنید (لایسنس)
DET_SIZE = (640, 640)
REQUEST_TIMEOUT = 10


class EnrollApp:
    def __init__(self, root):
        self.root = root
        self.root.title("ثبت چهره")
        self.root.geometry("1100x750")

        # --- وضعیت داخلی ---
        self.cap = None
        self.capture_thread = None
        self.capture_running = False
        self.live_frame = None          # آخرین فریم زنده (برای پیش‌نمایش)
        self.frame_lock = threading.Lock()

        self.frozen_frame = None        # فریمی که گرفته شده (take picture)
        self.person_boxes = []          # [(x1,y1,x2,y2), ...]
        self.selected_box_idx = None
        self.current_embedding = None   # امبدینگ چهره‌ی تازه انتخاب‌شده (قبل از افزودن به لیست)
        self.selected_face_crop = None  # تصویر چهره‌ی تازه انتخاب‌شده

        # لیست موقتِ چند عکس/امبدینگ برای همین شخص، قبل از ارسال نهایی
        self.pending_embeddings = []    # [np.array(512,), ...]
        self.pending_face_crop = None   # آخرین چهره گرفته‌شده، برای آپلود به‌عنوان تصویر رکورد

        self.cameras_map = {}           # name -> رکورد دوربین از کالکشن cameras

        # --- بارگذاری مدل‌ها (یک‌بار در استارتاپ) ---
        self._set_status("در حال بارگذاری مدل‌ها...")
        self.root.update()
        try:
            self.yolo_model = YOLO(YOLO_MODEL_PATH)
            self.face_handler = FaceAnalysis(
                root='.',
                name=FACE_MODEL_NAME,
                providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
            )
            self.face_handler.prepare(ctx_id=0, det_size=DET_SIZE)
        except Exception as e:
            messagebox.showerror("خطا در بارگذاری مدل", str(e))
            raise

        self._build_ui()
        self._set_status("آماده")
        self.load_cameras()

        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

    # -----------------------------------------------------------------
    # رابط کاربری
    # -----------------------------------------------------------------
    def _build_ui(self):
        top = ttk.Frame(self.root, padding=8)
        top.pack(side=tk.TOP, fill=tk.X)

        ttk.Label(top, text="دوربین:").pack(side=tk.LEFT, padx=4)
        self.camera_combo = ttk.Combobox(top, width=35, state="readonly")
        self.camera_combo.pack(side=tk.LEFT, padx=4)

        self.refresh_cameras_btn = ttk.Button(top, text="🔄 بارگذاری دوربین‌ها", command=self.load_cameras)
        self.refresh_cameras_btn.pack(side=tk.LEFT, padx=4)

        self.connect_btn = ttk.Button(top, text="اتصال به دوربین", command=self.toggle_camera)
        self.connect_btn.pack(side=tk.LEFT, padx=4)

        self.capture_btn = ttk.Button(top, text="گرفتن عکس", command=self.take_picture, state=tk.DISABLED)
        self.capture_btn.pack(side=tk.LEFT, padx=4)

        self.live_btn = ttk.Button(top, text="بازگشت به تصویر زنده", command=self.back_to_live, state=tk.DISABLED)
        self.live_btn.pack(side=tk.LEFT, padx=4)

        # --- ناحیه اصلی: تصویر + فرم ---
        main = ttk.Frame(self.root, padding=8)
        main.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        # بوم نمایش تصویر (سمت چپ)
        self.canvas = tk.Canvas(main, width=800, height=600, bg="black")
        self.canvas.pack(side=tk.LEFT, padx=4, pady=4)
        self.canvas.bind("<Button-1>", self._on_canvas_click)

        # فرم اطلاعات (سمت راست)
        form = ttk.Frame(main, padding=8)
        form.pack(side=tk.LEFT, fill=tk.Y, padx=8)

        ttk.Label(form, text="چهره انتخاب‌شده:").pack(anchor="e")
        self.face_preview_label = ttk.Label(form)
        self.face_preview_label.pack(pady=4)

        ttk.Label(form, text="نام:").pack(anchor="e", pady=(12, 0))
        self.name_entry = ttk.Entry(form, width=30)
        self.name_entry.pack(pady=2)

        ttk.Label(form, text="سن:").pack(anchor="e", pady=(8, 0))
        self.age_entry = ttk.Entry(form, width=30)
        self.age_entry.pack(pady=2)

        ttk.Label(form, text="جنسیت:").pack(anchor="e", pady=(8, 0))
        self.gender_combo = ttk.Combobox(form, values=["مرد", "زن"], state="readonly", width=27)
        self.gender_combo.pack(pady=2)

        ttk.Label(form, text="نقش:").pack(anchor="e", pady=(8, 0))
        self.role_combo = ttk.Combobox(form, values=["مجاز", "غیر مجاز"], state="readonly", width=27)
        self.role_combo.pack(pady=2)

        ttk.Label(form, text="کد ملی:").pack(anchor="e", pady=(8, 0))
        self.social_entry = ttk.Entry(form, width=30)
        self.social_entry.pack(pady=2)

        self.add_btn = ttk.Button(form, text="➕ افزودن این چهره به لیست", command=self.add_to_pending, state=tk.DISABLED)
        self.add_btn.pack(pady=(20, 4))

        self.pending_label = ttk.Label(form, text="عکس‌های اضافه‌شده: ۰", anchor="e")
        self.pending_label.pack(pady=2)

        self.clear_pending_btn = ttk.Button(form, text="پاک کردن لیست عکس‌ها", command=self.clear_pending, state=tk.DISABLED)
        self.clear_pending_btn.pack(pady=2)

        self.send_btn = ttk.Button(form, text="ارسال نهایی به پایگاه داده", command=self.send_to_db, state=tk.DISABLED)
        self.send_btn.pack(pady=(16, 4))

        self.status_label = ttk.Label(self.root, text="", anchor="w", relief=tk.SUNKEN)
        self.status_label.pack(side=tk.BOTTOM, fill=tk.X)

    def _set_status(self, text):
        if hasattr(self, "status_label"):
            self.status_label.config(text=text)
        else:
            print(text)

    # -----------------------------------------------------------------
    # دوربین / RTSP
    # -----------------------------------------------------------------
    def load_cameras(self):
        """بارگذاری لیست دوربین‌ها از کالکشن cameras در PocketBase"""
        url = f"{POCKETBASE_URL}/api/collections/cameras/records"
        try:
            res = requests.get(url, params={"perPage": 200}, timeout=REQUEST_TIMEOUT)
            res.raise_for_status()
            records = res.json().get("items", [])
        except Exception as e:
            messagebox.showerror("خطا", f"بارگذاری دوربین‌ها ناموفق بود:\n{e}")
            return

        self.cameras_map = {}
        names = []
        for rec in records:
            name = rec.get("name") or rec.get("id")
            self.cameras_map[name] = rec
            names.append(name)

        self.camera_combo.config(values=names)
        if names:
            self.camera_combo.current(0)
        self._set_status(f"{len(names)} دوربین بارگذاری شد")

    def _build_rtsp_url(self, cam):
        """ساخت آدرس RTSP از روی رکورد دوربین.
        اگر خود فیلد rtspUrl پر و کامل باشد همان استفاده می‌شود،
        در غیر این‌صورت از ip/port/username/password/rtspName ساخته می‌شود."""
        rtsp_url = (cam.get("rtspUrl") or "").strip()
        if rtsp_url:
            return rtsp_url

        ip = (cam.get("ip") or "").strip()
        port = (cam.get("port") or "").strip()
        rtsp_name = (cam.get("rtspName") or "").strip()
        username = (cam.get("username") or "").strip()
        password = (cam.get("password") or "").strip()

        if not ip:
            return None

        auth = f"{username}:{password}@" if username else ""
        port_part = f":{port}" if port else ""
        path_part = f"/{rtsp_name}" if rtsp_name else ""

        return f"rtsp://{auth}{ip}{port_part}{path_part}"

    def toggle_camera(self):
        if self.capture_running:
            self._stop_camera()
            self.connect_btn.config(text="اتصال به دوربین")
            self.capture_btn.config(state=tk.DISABLED)
        else:
            cam_name = self.camera_combo.get().strip()
            if not cam_name or cam_name not in self.cameras_map:
                messagebox.showwarning("خطا", "یک دوربین از لیست انتخاب کنید")
                return

            cam = self.cameras_map[cam_name]
            url = self._build_rtsp_url(cam)
            if not url:
                messagebox.showwarning("خطا", "برای این دوربین آدرس معتبری ساخته نشد (فیلد ip یا rtspUrl خالی است)")
                return

            self._start_camera(url)
            self.connect_btn.config(text="قطع اتصال")
            self.capture_btn.config(state=tk.NORMAL)

    def _start_camera(self, url):
        self.cap = cv2.VideoCapture(url)
        if not self.cap.isOpened():
            messagebox.showerror("خطا", "اتصال به دوربین برقرار نشد")
            self.cap = None
            return

        self.capture_running = True
        self.capture_thread = threading.Thread(target=self._capture_loop, daemon=True)
        self.capture_thread.start()
        self._update_preview()
        self._set_status("دوربین متصل شد")

    def _stop_camera(self):
        self.capture_running = False
        if self.capture_thread:
            self.capture_thread.join(timeout=1)
        if self.cap:
            self.cap.release()
            self.cap = None
        self._set_status("دوربین قطع شد")

    def _capture_loop(self):
        while self.capture_running and self.cap is not None:
            ret, frame = self.cap.read()
            if not ret or frame is None:
                time.sleep(0.05)
                continue
            with self.frame_lock:
                self.live_frame = frame
            time.sleep(0.01)

    def _update_preview(self):
        if not self.capture_running:
            return
        # فقط وقتی هنوز روی حالت "عکس گرفته‌شده" نیستیم، تصویر زنده را نشان بده
        if self.frozen_frame is None:
            with self.frame_lock:
                frame = None if self.live_frame is None else self.live_frame.copy()
            if frame is not None:
                self._show_on_canvas(frame)
        self.root.after(33, self._update_preview)

    def back_to_live(self):
        self.frozen_frame = None
        self.person_boxes = []
        self.selected_box_idx = None
        self.current_embedding = None
        self.selected_face_crop = None
        self.live_btn.config(state=tk.DISABLED)
        self.send_btn.config(state=tk.DISABLED)
        self.face_preview_label.config(image="")

    # -----------------------------------------------------------------
    # گرفتن عکس + تشخیص افراد
    # -----------------------------------------------------------------
    def take_picture(self):
        with self.frame_lock:
            frame = None if self.live_frame is None else self.live_frame.copy()

        if frame is None:
            messagebox.showwarning("خطا", "هنوز تصویری از دوربین دریافت نشده")
            return

        self.frozen_frame = frame
        self.selected_box_idx = None
        self.current_embedding = None
        self.selected_face_crop = None
        self.send_btn.config(state=tk.DISABLED)
        self.face_preview_label.config(image="")

        try:
            result = self.yolo_model.predict(frame, classes=[0], verbose=False)[0]
        except Exception as e:
            messagebox.showerror("خطا در تشخیص", str(e))
            return

        self.person_boxes = []
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            self.person_boxes.append((x1, y1, x2, y2))

        if not self.person_boxes:
            messagebox.showinfo("توجه", "هیچ فردی در تصویر شناسایی نشد")

        self.live_btn.config(state=tk.NORMAL)
        self._draw_boxes_and_show()
        self._set_status(f"{len(self.person_boxes)} فرد شناسایی شد — روی فرد مورد نظر کلیک کنید")

    def _draw_boxes_and_show(self):
        if self.frozen_frame is None:
            return
        display = self.frozen_frame.copy()
        for i, (x1, y1, x2, y2) in enumerate(self.person_boxes):
            color = (0, 255, 0) if i != self.selected_box_idx else (0, 0, 255)
            thickness = 2 if i != self.selected_box_idx else 3
            cv2.rectangle(display, (x1, y1), (x2, y2), color, thickness)
            cv2.putText(display, str(i + 1), (x1, max(y1 - 8, 0)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        self._show_on_canvas(display)

    def _show_on_canvas(self, bgr_frame):
        rgb = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
        h, w = rgb.shape[:2]
        canvas_w, canvas_h = 800, 600
        scale = min(canvas_w / w, canvas_h / h)
        new_w, new_h = int(w * scale), int(h * scale)
        resized = cv2.resize(rgb, (new_w, new_h))

        img = Image.fromarray(resized)
        self._tk_img = ImageTk.PhotoImage(image=img)  # جلوگیری از garbage collection
        self.canvas.delete("all")
        self.canvas.create_image(canvas_w // 2, canvas_h // 2, image=self._tk_img)

        # برای تبدیل مختصات کلیک بعداً لازم داریم
        self._display_scale = scale
        self._display_offset = ((canvas_w - new_w) // 2, (canvas_h - new_h) // 2)

    # -----------------------------------------------------------------
    # انتخاب فرد با کلیک + استخراج چهره
    # -----------------------------------------------------------------
    def _on_canvas_click(self, event):
        if self.frozen_frame is None or not self.person_boxes:
            return

        ox, oy = self._display_offset
        scale = self._display_scale
        # تبدیل مختصات کلیک روی canvas به مختصات فریم اصلی
        fx = (event.x - ox) / scale
        fy = (event.y - oy) / scale

        for i, (x1, y1, x2, y2) in enumerate(self.person_boxes):
            if x1 <= fx <= x2 and y1 <= fy <= y2:
                self._select_person(i)
                return

    def _select_person(self, idx):
        self.selected_box_idx = idx
        self._draw_boxes_and_show()

        x1, y1, x2, y2 = self.person_boxes[idx]
        crop = self.frozen_frame[y1:y2, x1:x2]
        if crop.size == 0:
            messagebox.showwarning("خطا", "ناحیه انتخاب‌شده نامعتبر است")
            return

        try:
            faces = self.face_handler.get(crop)
        except Exception as e:
            messagebox.showerror("خطا در استخراج چهره", str(e))
            return

        if not faces:
            messagebox.showwarning("خطا", "چهره‌ای در این ناحیه یافت نشد")
            self.current_embedding = None
            self.selected_face_crop = None
            self.add_btn.config(state=tk.DISABLED)
            self.face_preview_label.config(image="")
            return

        face = faces[0]
        self.current_embedding = face.embedding

        fx1, fy1, fx2, fy2 = map(int, face.bbox)
        fx1, fy1 = max(fx1, 0), max(fy1, 0)
        fx2, fy2 = min(fx2, crop.shape[1]), min(fy2, crop.shape[0])
        face_crop = crop[fy1:fy2, fx1:fx2]

        if face_crop.size == 0:
            face_crop = crop  # fallback

        self.selected_face_crop = face_crop.copy()
        self._show_face_preview(face_crop)
        self.add_btn.config(state=tk.NORMAL)
        self._set_status("چهره استخراج شد — می‌توانید آن را به لیست اضافه کنید")

    # -----------------------------------------------------------------
    # مدیریت لیست موقتِ چند عکس برای یک نفر
    # -----------------------------------------------------------------
    def add_to_pending(self):
        if self.current_embedding is None:
            return

        self.pending_embeddings.append(self.current_embedding)
        self.pending_face_crop = self.selected_face_crop  # آخرین چهره برای آپلود نگه داشته می‌شود

        self.pending_label.config(text=f"عکس‌های اضافه‌شده: {len(self.pending_embeddings)}")
        self.clear_pending_btn.config(state=tk.NORMAL)
        self.send_btn.config(state=tk.NORMAL)

        # آماده‌سازی برای گرفتن عکس بعدی از همان شخص —
        # فرم (نام/سن/...) دست‌نخورده می‌ماند چون شخص همان است
        self.current_embedding = None
        self.selected_face_crop = None
        self.add_btn.config(state=tk.DISABLED)
        self.face_preview_label.config(image="")
        self.frozen_frame = None
        self.person_boxes = []
        self.selected_box_idx = None
        self.live_btn.config(state=tk.DISABLED)

        self._set_status(
            f"اضافه شد ({len(self.pending_embeddings)} عکس در لیست) — "
            "برای عکس بعدی «گرفتن عکس» را بزنید یا اطلاعات را تکمیل و ارسال کنید"
        )

    def clear_pending(self):
        """پاک کردن لیست از دکمه — با تأییدیه."""
        if not self.pending_embeddings:
            return
        if not messagebox.askyesno("تأیید", "لیست عکس‌های اضافه‌شده پاک شود؟"):
            return
        self._reset_pending()

    def _reset_pending(self):
        """پاک کردن داخلی لیست، بدون تأییدیه (بعد از ارسال موفق فراخوانی می‌شود)."""
        self.pending_embeddings = []
        self.pending_face_crop = None
        self.pending_label.config(text="عکس‌های اضافه‌شده: ۰")
        self.clear_pending_btn.config(state=tk.DISABLED)
        self.send_btn.config(state=tk.DISABLED)

    def _show_face_preview(self, bgr_face):
        rgb = cv2.cvtColor(bgr_face, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(rgb).resize((160, 160))
        self._tk_face_img = ImageTk.PhotoImage(image=img)
        self.face_preview_label.config(image=self._tk_face_img)

    # -----------------------------------------------------------------
    # ارسال به PocketBase
    # -----------------------------------------------------------------
    def send_to_db(self):
        if not self.pending_embeddings:
            messagebox.showwarning("خطا", "حداقل یک چهره باید به لیست اضافه شده باشد")
            return

        name = self.name_entry.get().strip()
        if not name:
            messagebox.showwarning("خطا", "وارد کردن نام الزامی است")
            return

        age = self.age_entry.get().strip()
        gender = self.gender_combo.get().strip()
        role_fa = self.role_combo.get().strip()
        role_map = {"مجاز": "approve", "غیر مجاز": "denied"}
        role = role_map.get(role_fa, "")
        social = self.social_entry.get().strip()

        # همه‌ی امبدینگ‌های جمع‌شده (چند عکس) با هم به یک لیست تخت تبدیل می‌شوند —
        # دقیقاً همان قالبی که load_embeddings_from_db در سمت اصلی انتظار دارد
        # (چند بردار ۵۱۲تایی پشت سر هم).
        embedding_list = []
        for emb in self.pending_embeddings:
            embedding_list.extend(float(v) for v in emb.tolist())

        count = len(self.pending_embeddings)

        try:
            existing = self._find_existing_record(name)
            if existing:
                self._append_embedding(existing, embedding_list, age, gender, role, social)
                messagebox.showinfo("موفق", f"{count} امبدینگ جدید به «{name}» اضافه شد")
            else:
                self._create_record(name, embedding_list, age, gender, role, social)
                messagebox.showinfo("موفق", f"«{name}» با {count} عکس ثبت شد")

            self._set_status("ارسال با موفقیت انجام شد")
            self.back_to_live()
            self._reset_pending()
            self.name_entry.delete(0, tk.END)
            self.age_entry.delete(0, tk.END)
            self.role_combo.set("")
            self.social_entry.delete(0, tk.END)
            self.gender_combo.set("")

        except Exception as e:
            messagebox.showerror("خطا در ارسال", str(e))

    def _find_existing_record(self, name):
        url = f"{POCKETBASE_URL}/api/collections/{COLLECTION}/records"
        # اسم را داخل فیلتر escape می‌کنیم
        safe_name = name.replace('"', '\\"')
        params = {"filter": f'name="{safe_name}"', "perPage": 1}
        res = requests.get(url, params=params, timeout=REQUEST_TIMEOUT)
        res.raise_for_status()
        items = res.json().get("items", [])
        return items[0] if items else None

    def _create_record(self, name, embedding_list, age, gender, role, social):
        url = f"{POCKETBASE_URL}/api/collections/{COLLECTION}/records"
        data = {
            "name": name,
            "embdanings": json.dumps(embedding_list),
            "age": age,
            "gender": gender,
            "role": role,
            "socialnumber": social,
        }
        files = self._face_file_payload()
        res = requests.post(url, data=data, files=files, timeout=REQUEST_TIMEOUT)
        res.raise_for_status()

    def _append_embedding(self, existing_record, new_embedding_list, age, gender, role, social):
        record_id = existing_record["id"]
        old_embeddings = existing_record.get("embdanings") or []
        if not isinstance(old_embeddings, list):
            old_embeddings = []
        merged = old_embeddings + new_embedding_list

        url = f"{POCKETBASE_URL}/api/collections/{COLLECTION}/records/{record_id}"
        data = {"embdanings": json.dumps(merged)}
        # فقط اگر فیلدی پر شده باشد آپدیتش کن (فیلد خالی را رونویسی نکن)
        if age:
            data["age"] = age
        if gender:
            data["gender"] = gender
        if role:
            data["role"] = role
        if social:
            data["socialnumber"] = social

        files = self._face_file_payload()
        res = requests.patch(url, data=data, files=files, timeout=REQUEST_TIMEOUT)
        res.raise_for_status()

    def _face_file_payload(self):
        if self.pending_face_crop is None:
            return None
        ok, buf = cv2.imencode(".jpg", self.pending_face_crop)
        if not ok:
            return None
        return {"image": ("face.jpg", io.BytesIO(buf.tobytes()), "image/jpeg")}

    # -----------------------------------------------------------------
    def _on_close(self):
        self._stop_camera()
        self.root.destroy()


if __name__ == "__main__":
    root = tk.Tk()
    app = EnrollApp(root)
    root.mainloop()