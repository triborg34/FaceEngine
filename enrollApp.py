# -*- coding: utf-8 -*-


import io
import json
import re
import threading
import time
from datetime import date

import cv2
import numpy as np
import requests
import tkinter as tk
import tkinter.font as tkfont
from tkinter import ttk, messagebox, filedialog
from PIL import Image, ImageTk

from ultralytics import YOLO
from insightface.app import FaceAnalysis

# ---------------------------------------------------------------------------
# تنظیمات قابل تغییر
# ---------------------------------------------------------------------------
POCKETBASE_URL = "http://127.0.0.1:8091"
COLLECTION = "known_face"
# شناسه‌ی کالکشن در PocketBase (برای ساخت آدرس فایل‌ها استفاده می‌شود؛
# از پنل ادمین PocketBase قابل مشاهده است. با تغییر نام/شناسه باید به‌روز شود)
COLLECTION_ID = "pbc_745878446"
YOLO_MODEL_PATH = "models/yolov8n.pt"
# برای استفاده تجاری از antelopev2 استفاده نکنید (لایسنس)
FACE_MODEL_NAME = "buffalo_l"
DET_SIZE = (640, 640)
REQUEST_TIMEOUT = 10

# تعداد مؤلفه‌های هر بردار امبدینگ (برای شمارش تعداد عکس‌های هر فرد)
EMBEDDING_DIM = 512

# نگاشت برچسب‌های فارسی رابط به مقادیر انگلیسی ذخیره‌شده در دیتابیس
GENDER_MAP = {"مرد": "male", "زن": "female"}
ROLE_MAP = {"مجاز": "approve", "غیر مجاز": "denied"}
# نگاشت معکوس، برای نمایش مقدار دیتابیس در فرم ویرایش
GENDER_FA = {v: k for k, v in GENDER_MAP.items()}
ROLE_FA = {v: k for k, v in ROLE_MAP.items()}

# ---------------------------------------------------------------------------
# پالت رنگ‌ها و ظاهر برنامه
# ---------------------------------------------------------------------------
C_PRIMARY = "#2563eb"          # آبی اصلی
C_PRIMARY_DARK = "#1d4ed8"     # هاور
C_PRIMARY_SOFT = "#dbeafe"     # پس‌زمینه‌ی ملایم آبی
C_SUCCESS = "#16a34a"          # سبز دکمه‌ی ارسال
C_SUCCESS_DARK = "#15803d"
C_BG = "#f1f5f9"              # پس‌زمینه‌ی اصلی
C_SURFACE = "#ffffff"          # کارت‌ها
C_BORDER = "#e2e8f0"           # حاشیه‌ها
C_TEXT = "#0f172a"             # متن اصلی
C_MUTED = "#64748b"            # متن ثانویه
C_DARK_VIEW = "#0f172a"        # پس‌زماینه بوم تصویر
FONT_CANDIDATES = ("Vazirmatn", "IRANSansX", "IRANSans",
                   "B Nazanin", "Tahoma", "Segoe UI")

# ---------------------------------------------------------------------------
# توابع تبدیل تاریخ شمسی (بدون وابستگی خارجی)
# ---------------------------------------------------------------------------
_PERSIAN_DIGITS = str.maketrans("۰۱۲۳۴۵۶۷۸۹٠١٢٣٤٥٦٧٨٩", "0123456789" * 2)


def jalali_to_gregorian(jy, jm, jd):
    """تبدیل تاریخ شمسی به میلادی -> (gy, gm, gd)"""
    jy += 1595
    days = -355668 + 365 * jy + (jy // 33) * 8 + (((jy % 33) + 3) // 4) + jd
    if jm < 7:
        days += (jm - 1) * 31
    else:
        days += (jm - 7) * 30 + 186

    gy = 400 * (days // 146097)
    days %= 146097
    if days > 36524:
        days -= 1
        gy += 100 * (days // 36524)
        days %= 36524
        if days >= 365:
            days += 1
    gy += 4 * (days // 1461)
    days %= 1461
    if days > 365:
        gy += (days - 1) // 365
        days = (days - 1) % 365
    gd = days + 1

    leap = (gy % 4 == 0 and gy % 100 != 0) or gy % 400 == 0
    month_days = (0, 31, 29 if leap else 28, 31, 30, 31, 30,
                  31, 31, 30, 31, 30, 31)
    gm = 1
    while gm < 13 and gd > month_days[gm]:
        gd -= month_days[gm]
        gm += 1
    return gy, gm, gd


def gregorian_to_jalali(gy, gm, gd):
    """تبدیل تاریخ میلادی به شمسی -> (jy, jm, jd)"""
    g_d_m = (0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334)
    jy = 979 if gy > 1600 else 0
    gy -= 1600 if gy > 1600 else 621
    gy2 = gy + 1 if gm > 2 else gy
    days = (365 * gy + (gy2 + 3) // 4 - (gy2 + 99) // 100
            + (gy2 + 399) // 400 - 80 + gd + g_d_m[gm - 1])

    jy += 33 * (days // 12053)
    days %= 12053
    jy += 4 * (days // 1461)
    days %= 1461
    if days > 365:
        jy += (days - 1) // 365
        days = (days - 1) % 365

    if days < 186:
        jm = 1 + days // 31
        jd = 1 + days % 31
    else:
        jm = 7 + (days - 186) // 30
        jd = 1 + (days - 186) % 30
    return jy, jm, jd


def parse_jalali_date(text):
    """خواندن متن مثل '1377/04/12' (ارقام فارسی/عربی هم پذیرفته می‌شود)
    و برگرداندن tuple شمسی، یا None اگر نامعتبر باشد."""
    if not text:
        return None
    text = text.translate(_PERSIAN_DIGITS).strip()
    parts = [p for p in re.split(r"[/\-.]", text) if p]
    if len(parts) != 3:
        return None
    try:
        jy, jm, jd = (int(p) for p in parts)
    except ValueError:
        return None
    if not (1200 <= jy <= 1600 and 1 <= jm <= 12 and 1 <= jd <= 31):
        return None
    # اعتبارسنجی (مثلاً ۳۰ اسفند در سال کبیسه‌الحقبودن): تبدیل رفت و برگشت
    gy, gm, gd = jalali_to_gregorian(jy, jm, jd)
    if gregorian_to_jalali(gy, gm, gd) != (jy, jm, jd):
        return None
    return jy, jm, jd


def jalali_age(jalali_birth, today=None):
    """سن به سال تمام‌شده برای تولد شمسی."""
    today = today or date.today()
    gy, gm, gd = jalali_to_gregorian(*jalali_birth)
    birth = date(gy, gm, gd)
    if birth > today:
        return -1
    return today.year - birth.year - ((today.month, today.day) < (birth.month, birth.day))


def fa(value):
    """تبدیل ارقام لاتین به فارسی برای نمایش در رابط."""
    return str(value).translate(str.maketrans("0123456789", "۰۱۲۳۴۵۶۷۸۹"))


class EnrollApp:
    def __init__(self, root):
        self.root = root
        self.root.title("ثبت چهره — Face Enrollment")
        self.root.configure(bg=C_BG)

        # فونت فارسیِ موجود در سیستم
        self.font = self._pick_font()

        # وسط‌چین کردن پنجره
        win_w, win_h = 1180, 860
        self.root.geometry(
            f"{win_w}x{win_h}"
            f"+{max((root.winfo_screenwidth() - win_w) // 2, 0)}"
            f"+{max((root.winfo_screenheight() - win_h) // 2, 0)}"
        )
        self.root.minsize(1040, 720)

        # --- وضعیت داخلی ---
        self.cap = None
        self.capture_thread = None
        self.capture_running = False
        self.live_frame = None          # آخرین فریم زنده (برای پیش‌نمایش)
        self.frame_lock = threading.Lock()

        self.frozen_frame = None        # فریمی که گرفته شده (take picture)
        self.person_boxes = []          # [(x1,y1,x2,y2), ...]
        self.selected_box_idx = None
        # امبدینگ چهره‌ی تازه انتخاب‌شده (قبل از افزودن به لیست)
        self.current_embedding = None
        self.selected_face_crop = None  # تصویر چهره‌ی تازه انتخاب‌شده

        # لیست موقتِ چند عکس/امبدینگ برای همین شخص، قبل از ارسال نهایی
        self.pending_embeddings = []    # [np.array(512,), ...]
        self.pending_crops = []         # تصویر چهره‌ی هر آیتم (برای پیش‌نمایش)
        # آخرین چهره گرفته‌شده، برای آپلود به‌عنوان تصویر رکورد
        self.pending_face_crop = None
        self._thumb_imgs = []           # نگه‌داری PhotoImage ها (جلوگیری از GC)

        self.cameras_map = {}           # name -> رکورد دوربین از کالکشن cameras
        self.people_map = {}            # record_id -> رکورد شخص از کالکشن known_face
        self._people_loaded = False     # آیا فهرست افراد یک‌بار بارگذاری شده است
        self._people_photo = None       # نگه‌داری PhotoImage تصویر رکورد

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
    def _pick_font(self):
        """انتخاب فونت فارسیِ نصب‌شده روی سیستم."""
        try:
            families = set(tkfont.families(self.root))
        except tk.TclError:
            families = set()
        for name in FONT_CANDIDATES:
            if name in families:
                return name
        return "TkDefaultFont"

    def _setup_styles(self):
        """ظاهر مدرن برای ویجت‌های ttk."""
        style = ttk.Style(self.root)
        try:
            style.theme_use("clam")
        except tk.TclError:
            pass

        f = self.font
        style.configure(".", font=(f, 10), background=C_BG, foreground=C_TEXT)

        # قاب‌ها
        style.configure("TFrame", background=C_BG)
        style.configure("Card.TFrame", background=C_SURFACE)

        # متن‌ها
        style.configure("TLabel", background=C_BG, foreground=C_TEXT)
        style.configure("Card.TLabel", background=C_SURFACE, foreground=C_TEXT)
        style.configure("Card.Muted.TLabel",
                        background=C_SURFACE, foreground=C_MUTED)
        style.configure(
            "Age.TLabel", background=C_PRIMARY_SOFT, foreground=C_PRIMARY,
            font=(f, 10, "bold"), padding=(8, 5), anchor="center",
        )
        style.configure(
            "Counter.TLabel", background=C_SURFACE, foreground=C_PRIMARY,
            font=(f, 10, "bold"), anchor="w",
        )

        # دکمه‌ی معمولی
        style.configure(
            "TButton", font=(f, 10), padding=(14, 7), relief="flat",
            background="#f8fafc", foreground=C_TEXT,
            bordercolor="#cbd5e1", lightcolor="#f8fafc", darkcolor="#f8fafc",
        )
        style.map(
            "TButton",
            background=[("active", C_BG), ("pressed", "#e2e8f0"),
                        ("disabled", C_BG)],
            foreground=[("disabled", "#a5b4c8")],
        )

        # دکمه‌ی اصلی (آبی)
        style.configure(
            "Primary.TButton", font=(f, 10, "bold"), padding=(14, 6), relief="flat",
            background=C_PRIMARY, foreground="white",
            bordercolor=C_PRIMARY, lightcolor=C_PRIMARY, darkcolor=C_PRIMARY,
        )
        style.map(
            "Primary.TButton",
            background=[("active", C_PRIMARY_DARK), ("pressed",
                                                     "#1e40af"), ("disabled", "#bfdbfe")],
            foreground=[("disabled", "#eff6ff")],
        )

        # دکمه‌ی ارسال (سبز)
        style.configure(
            "Success.TButton", font=(f, 11, "bold"), padding=(14, 7), relief="flat",
            background=C_SUCCESS, foreground="white",
            bordercolor=C_SUCCESS, lightcolor=C_SUCCESS, darkcolor=C_SUCCESS,
        )
        style.map(
            "Success.TButton",
            background=[("active", C_SUCCESS_DARK), ("pressed",
                                                     "#166534"), ("disabled", "#bbf7d0")],
            foreground=[("disabled", "#f0fdf4")],
        )

        # دکمه‌ی کم‌رنگ
        style.configure(
            "Ghost.TButton", font=(f, 9), padding=(10, 5), relief="flat",
            background="#f8fafc", foreground=C_MUTED,
            bordercolor="#cbd5e1", lightcolor="#f8fafc", darkcolor="#f8fafc",
        )
        style.map(
            "Ghost.TButton",
            background=[("active", C_BG), ("pressed", "#e2e8f0"),
                        ("disabled", C_BG)],
            foreground=[("disabled", "#a5b4c8")],
        )

        # دکمه‌ی روی هدر آبی
        style.configure(
            "OnHeader.TButton", font=(f, 10), padding=(14, 7), relief="flat",
            background=C_PRIMARY_DARK, foreground="#dbeafe",
            bordercolor="#60a5fa", lightcolor=C_PRIMARY_DARK, darkcolor=C_PRIMARY_DARK,
        )
        style.map(
            "OnHeader.TButton",
            background=[("active", "#1e40af"), ("pressed", "#1e3a8a")],
            foreground=[("active", "white")],
        )

        # دکمه‌ی حذف (قرمز)
        style.configure(
            "Danger.TButton", font=(f, 10, "bold"), padding=(14, 6), relief="flat",
            background="#fee2e2", foreground="#dc2626",
            bordercolor="#fecaca", lightcolor="#fee2e2", darkcolor="#fee2e2",
        )
        style.map(
            "Danger.TButton",
            background=[("active", "#fecaca"), ("pressed", "#fca5a5"),
                        ("disabled", C_SURFACE)],
            foreground=[("disabled", "#fca5a5")],
        )

        # فیلدهای ورودی
        style.configure(
            "TEntry", fieldbackground="white", foreground=C_TEXT, insertcolor=C_TEXT,
            bordercolor=C_BORDER, lightcolor=C_BORDER, darkcolor=C_BORDER,
            padding=6, relief="flat",
        )
        style.map(
            "TEntry",
            bordercolor=[("focus", C_PRIMARY)],
            lightcolor=[("focus", C_PRIMARY)],
            darkcolor=[("focus", C_PRIMARY)],
        )

        # لیست‌کشویی
        style.configure(
            "TCombobox", fieldbackground="white", background="white",
            foreground=C_TEXT, bordercolor=C_BORDER, arrowcolor=C_MUTED,
            padding=6, relief="flat",
        )
        style.map(
            "TCombobox",
            fieldbackground=[("readonly", "white")],
            foreground=[("readonly", C_TEXT)],
            bordercolor=[("focus", C_PRIMARY)],
            lightcolor=[("focus", C_PRIMARY)],
            darkcolor=[("focus", C_PRIMARY)],
            arrowcolor=[("active", C_PRIMARY)],
        )

        style.configure("TSeparator", background=C_BORDER)

        # تب‌ها
        style.configure(
            "TNotebook", background=C_BG, borderwidth=0, tabmargins=(0, 0, 0, 0),
        )
        style.configure(
            "TNotebook.Tab", font=(f, 10), padding=(22, 8),
            background="#e2e8f0", foreground=C_MUTED,
        )
        style.map(
            "TNotebook.Tab",
            background=[("selected", C_SURFACE)],
            foreground=[("selected", C_PRIMARY)],
        )

        # جدول فهرست افراد
        style.configure(
            "People.Treeview", font=(f, 10), background=C_SURFACE,
            fieldbackground=C_SURFACE, foreground=C_TEXT,
            rowheight=28, bordercolor=C_BORDER, relief="flat",
        )
        style.configure(
            "People.Treeview.Heading", font=(f, 9, "bold"),
            background=C_BG, foreground=C_MUTED, relief="flat", padding=(6, 7),
        )
        style.map(
            "People.Treeview",
            background=[("selected", C_PRIMARY_SOFT)],
            foreground=[("selected", C_PRIMARY)],
        )
        style.map(
            "People.Treeview.Heading",
            background=[("active", "#e2e8f0")],
        )

    def _build_ui(self):
        self._setup_styles()

        # ---------------- هدر ----------------
        header = tk.Frame(self.root, bg=C_PRIMARY)
        header.pack(side=tk.TOP, fill=tk.X)
        tk.Label(
            header, text="ثبت چهره", bg=C_PRIMARY, fg="white",
            font=(self.font, 16, "bold"),
        ).pack(side=tk.RIGHT, padx=(0, 18), pady=12)
        tk.Label(
            header, text="Face Enrollment", bg=C_PRIMARY, fg="#bfd7ff",
            font=(self.font, 10),
        ).pack(side=tk.RIGHT, pady=12)
        self.refresh_cameras_btn = ttk.Button(
            header, text="بروزرسانی دوربین‌ها", command=self.load_cameras,
            style="OnHeader.TButton",
        )
        self.refresh_cameras_btn.pack(side=tk.LEFT, padx=16, pady=10)

        # ---------------- تب‌ها ----------------
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        self.notebook.bind("<<NotebookTabChanged>>", self._on_tab_changed)

        enroll_tab = tk.Frame(self.notebook, bg=C_BG)
        self.notebook.add(enroll_tab, text="ثبت چهره")
        self.manage_tab = tk.Frame(self.notebook, bg=C_BG)
        self.notebook.add(self.manage_tab, text="مدیریت افراد")

        # ---------------- نوار ابزار ----------------
        toolbar = tk.Frame(enroll_tab, bg=C_SURFACE)
        toolbar.pack(side=tk.TOP, fill=tk.X)
        ttk.Separator(enroll_tab, orient="horizontal").pack(
            side=tk.TOP, fill=tk.X)

        ttk.Label(toolbar, text="دوربین:", style="Card.TLabel").pack(
            side=tk.LEFT, padx=(16, 6), pady=10
        )
        self.camera_combo = ttk.Combobox(toolbar, width=34, state="readonly")
        self.camera_combo.pack(side=tk.LEFT, pady=10)

        self.connect_btn = ttk.Button(
            toolbar, text="اتصال به دوربین", command=self.toggle_camera,
            style="Primary.TButton",
        )
        self.connect_btn.pack(side=tk.LEFT, padx=(16, 6), pady=10)

        self.capture_btn = ttk.Button(
            toolbar, text="گرفتن عکس", command=self.take_picture, state=tk.DISABLED
        )
        self.capture_btn.pack(side=tk.LEFT, padx=6, pady=10)

        self.file_btn = ttk.Button(
            toolbar, text="عکس از فایل", command=self.load_picture_from_file
        )
        self.file_btn.pack(side=tk.LEFT, padx=6, pady=10)

        self.live_btn = ttk.Button(
            toolbar, text="بازگشت به تصویر زنده", command=self.back_to_live,
            state=tk.DISABLED,
        )
        self.live_btn.pack(side=tk.LEFT, padx=(6, 16), pady=10)

        # ---------------- ناحیه اصلی ----------------
        body = tk.Frame(enroll_tab, bg=C_BG)
        body.pack(fill=tk.BOTH, expand=True, padx=14, pady=14)

        # ستون چپ: کارت تصویر + نوار پیش‌نمایش چهره‌های اضافه‌شده
        left_col = tk.Frame(body, bg=C_BG)
        left_col.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 12))

        canvas_card = tk.Frame(
            left_col, bg=C_SURFACE, highlightbackground=C_BORDER, highlightthickness=1
        )
        canvas_card.pack(fill=tk.BOTH, expand=True)
        self.canvas = tk.Canvas(
            canvas_card, bg=C_DARK_VIEW, highlightthickness=0, bd=0)
        self.canvas.pack(fill=tk.BOTH, expand=True, padx=1, pady=1)
        self.canvas.bind("<Button-1>", self._on_canvas_click)

        # نوار پیش‌نمایش عکس‌های جمع‌شده (فقط وقتی چیزی اضافه شده باشد)
        self.thumbs_bar = tk.Frame(
            left_col, bg=C_SURFACE, highlightbackground=C_BORDER, highlightthickness=1
        )
        thumbs_head = tk.Frame(self.thumbs_bar, bg=C_SURFACE)
        thumbs_head.pack(fill=tk.X, padx=12, pady=(10, 0))
        tk.Label(
            thumbs_head, text="پیش‌نمایش چهره‌های اضافه‌شده قبل از ارسال",
            bg=C_SURFACE, fg=C_MUTED, font=(self.font, 9),
        ).pack(side=tk.RIGHT)
        self.thumbs_count = tk.Label(
            thumbs_head, bg=C_PRIMARY_SOFT, fg=C_PRIMARY,
            font=(self.font, 9, "bold"), padx=8,
        )
        self.thumbs_count.pack(side=tk.LEFT)

        self.thumbs_canvas = tk.Canvas(
            self.thumbs_bar, bg=C_SURFACE, highlightthickness=0, height=104
        )
        self.thumbs_canvas.pack(fill=tk.X, padx=12, pady=(6, 12))
        self.thumbs_inner = tk.Frame(self.thumbs_canvas, bg=C_SURFACE)
        self._thumbs_win = self.thumbs_canvas.create_window(
            (self.thumbs_canvas.winfo_width(), 0),
            window=self.thumbs_inner, anchor="ne",
        )
        self.thumbs_inner.bind("<Configure>", self._on_thumbs_configure)
        self.thumbs_canvas.bind("<Configure>", self._on_thumbs_configure)
        self.thumbs_canvas.bind("<MouseWheel>", self._on_thumbs_wheel)

        # فرم اطلاعات (سمت راست، چون رابط فارسی است)
        form = tk.Frame(
            body, bg=C_SURFACE, width=320,
            highlightbackground=C_BORDER, highlightthickness=1,
        )
        form.pack(side=tk.RIGHT, fill=tk.Y)
        form.pack_propagate(False)

        tk.Label(
            form, text="اطلاعات فرد", bg=C_SURFACE, fg=C_TEXT,
            font=(self.font, 12, "bold"),
        ).pack(anchor="e", padx=16, pady=(12, 0))
        ttk.Separator(form, orient="horizontal").pack(
            fill=tk.X, padx=16, pady=(6, 8))

        preview_box = tk.Frame(
            form, width=104, height=104, bg=C_BG,
            highlightbackground=C_BORDER, highlightthickness=1,
        )
        preview_box.pack(anchor="e", padx=16, pady=(2, 4))
        preview_box.pack_propagate(False)
        self.face_preview_label = tk.Label(
            preview_box, bg=C_BG, text="بدون چهره", fg="#94a3b8", font=(self.font, 9)
        )
        self.face_preview_label.pack(expand=True, fill=tk.BOTH)

        fields = tk.Frame(form, bg=C_SURFACE)
        fields.pack(fill=tk.X, padx=16, pady=(6, 0))

        # همه‌ی فیلدهای فرم به StringVar وصل می‌شوند تا ریست کردن فرم قابل
        # اتکا باشد؛ چون set("") روی Combobox با state="readonly" نمایش را
        # همیشه پاک نمی‌کند ولی مقدارِ متغیر همیشه پاک می‌شود.
        self.name_var = tk.StringVar()
        self.birth_var = tk.StringVar()
        self.gender_var = tk.StringVar()
        self.role_var = tk.StringVar()
        self.social_var = tk.StringVar()
        self.userwhom_var = tk.StringVar()

        def label_for(parent, text):
            tk.Label(
                parent, text=text, bg=C_SURFACE, fg=C_MUTED, font=(
                    self.font, 9)
            ).pack(anchor="e", pady=(7, 2))

        label_for(fields, "نام")
        self.name_entry = ttk.Entry(fields, textvariable=self.name_var)
        self.name_entry.pack(fill=tk.X)

        # عنوان تولد + نشان سن، هر دو در یک ردیف (برای جلوگیری از بلند شدن فرم)
        birth_row = tk.Frame(fields, bg=C_SURFACE)
        birth_row.pack(fill=tk.X, pady=(7, 2))
        tk.Label(
            birth_row, text="تاریخ تولد (شمسی)", bg=C_SURFACE, fg=C_MUTED,
            font=(self.font, 9),
        ).pack(side=tk.RIGHT)
        self.age_label = ttk.Label(birth_row, text="سن: —", style="Age.TLabel")
        self.age_label.pack(side=tk.LEFT)

        self.birth_entry = ttk.Entry(fields, textvariable=self.birth_var)
        self.birth_entry.pack(fill=tk.X)
        self.birth_entry.bind("<KeyRelease>", self._update_age_preview)
        tk.Label(
            fields, text="مثال: 1377/04/12", bg=C_SURFACE, fg="#94a3b8",
            font=(self.font, 8),
        ).pack(anchor="e")

        duo = tk.Frame(fields, bg=C_SURFACE)
        duo.pack(fill=tk.X)

        col_gender = tk.Frame(duo, bg=C_SURFACE)
        col_gender.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 4))
        label_for(col_gender, "جنسیت")
        self.gender_combo = ttk.Combobox(
            col_gender, textvariable=self.gender_var,
            values=["مرد", "زن"], state="readonly"
        )
        self.gender_combo.pack(fill=tk.X)

        col_role = tk.Frame(duo, bg=C_SURFACE)
        col_role.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(4, 0))
        label_for(col_role, "نقش")
        self.role_combo = ttk.Combobox(
            col_role, textvariable=self.role_var,
            values=["مجاز", "غیر مجاز"], state="readonly"
        )
        self.role_combo.pack(fill=tk.X)

        label_for(fields, "کد ملی")
        self.social_entry = ttk.Entry(fields, textvariable=self.social_var)
        self.social_entry.pack(fill=tk.X, pady=(0, 4))

        label_for(fields, "سمت")
        self.userwhom_combo = ttk.Combobox(
            fields, textvariable=self.userwhom_var,
            values=["ارباب رجوع", "همکار"], state="readonly"
        )
        self.userwhom_combo.pack(fill=tk.X, pady=(0, 4))


        

        # ---------------- دکمه‌ها (پایین فرم) ----------------
        actions = tk.Frame(form, bg=C_SURFACE)
        actions.pack(side=tk.BOTTOM, fill=tk.X, padx=16, pady=12)

        self.send_btn = ttk.Button(
            actions, text="ارسال نهایی به پایگاه داده", command=self.send_to_db,
            state=tk.DISABLED, style="Success.TButton",
        )
        self.send_btn.pack(fill=tk.X, pady=(0, 6))

        self.add_btn = ttk.Button(
            actions, text="افزودن این چهره به لیست", command=self.add_to_pending,
            state=tk.DISABLED, style="Primary.TButton",
        )
        self.add_btn.pack(fill=tk.X, pady=(0, 6))

        counter_row = tk.Frame(actions, bg=C_SURFACE)
        counter_row.pack(fill=tk.X)
        self.pending_label = ttk.Label(
            counter_row, text="عکس‌های اضافه‌شده: ۰", style="Counter.TLabel"
        )
        self.pending_label.pack(side=tk.LEFT)
        self.clear_pending_btn = ttk.Button(
            counter_row, text="پاک کردن", command=self.clear_pending,
            state=tk.DISABLED, style="Ghost.TButton",
        )
        self.clear_pending_btn.pack(side=tk.RIGHT)

        # ---------------- نوار وضعیت ----------------
        status_bar = tk.Frame(self.root, bg=C_SURFACE)
        status_bar.pack(side=tk.BOTTOM, fill=tk.X)
        ttk.Separator(self.root, orient="horizontal").pack(
            side=tk.BOTTOM, fill=tk.X)
        self.status_label = tk.Label(
            status_bar, text="", anchor="e", bg=C_SURFACE, fg=C_MUTED,
            font=(self.font, 9), padx=16, pady=7,
        )
        self.status_label.pack(fill=tk.X)

        # ---------------- تب مدیریت افراد ----------------
        self._build_manage_tab()

    def _set_status(self, text):
        if hasattr(self, "status_label"):
            self.status_label.config(text=text)
        else:
            print(text)

    def _update_age_preview(self, event=None):
        """نمایش زنده‌ی سن محاسبه‌شده از روی تاریخ تولد شمسی."""
        birth = parse_jalali_date(self.birth_var.get())
        if birth is None:
            self.age_label.config(text="سن: —")
            return
        age = jalali_age(birth)
        if age < 0:
            self.age_label.config(text="سن: — (تاریخ در آینده است)")
        else:
            self.age_label.config(text=f"سن: {age} سال")

    # -----------------------------------------------------------------
    # دوربین / RTSP
    # -----------------------------------------------------------------
    def load_cameras(self):
        """بارگذاری لیست دوربین‌ها از کالکشن cameras در PocketBase"""
        url = f"{POCKETBASE_URL}/api/collections/cameras/records"
        try:
            res = requests.get(
                url, params={"perPage": 200}, timeout=REQUEST_TIMEOUT)
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
                messagebox.showwarning(
                    "خطا", "برای این دوربین آدرس معتبری ساخته نشد (فیلد ip یا rtspUrl خالی است)")
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
        self.capture_thread = threading.Thread(
            target=self._capture_loop, daemon=True)
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
        if not self.pending_embeddings:
            self.send_btn.config(state=tk.DISABLED)
        self._clear_face_preview()
        if not self.capture_running:
            self.canvas.delete("all")
            self._set_status("آماده — می‌توانید عکس دیگری از فایل انتخاب کنید")

    # -----------------------------------------------------------------
    # گرفتن عکس + تشخیص افراد
    # -----------------------------------------------------------------
    def take_picture(self):
        with self.frame_lock:
            frame = None if self.live_frame is None else self.live_frame.copy()

        if frame is None:
            messagebox.showwarning("خطا", "هنوز تصویری از دوربین دریافت نشده")
            return

        self._freeze_frame(frame)
        if not self._detect_persons(frame):
            return

        if not self.person_boxes:
            messagebox.showinfo("توجه", "هیچ فردی در تصویر شناسایی نشد")

        self.live_btn.config(state=tk.NORMAL)
        self._draw_boxes_and_show()
        self._set_status(
            f"{len(self.person_boxes)} فرد شناسایی شد — روی فرد مورد نظر کلیک کنید")

    def load_picture_from_file(self):
        """انتخاب عکس از فایل به‌جای دوربین و اجرای همان فرایند تشخیص/ثبت."""
        path = filedialog.askopenfilename(
            title="انتخاب عکس",
            filetypes=[
                ("تصاویر", "*.jpg *.jpeg *.png *.bmp *.webp"),
                ("همه فایل‌ها", "*.*"),
            ],
        )
        if not path:
            return

        frame = cv2.imread(path)
        if frame is None:
            messagebox.showerror("خطا", f"خواندن تصویر ممکن نشد:\n{path}")
            return

        self._freeze_frame(frame)
        if not self._detect_persons(frame):
            return

        if not self.person_boxes:
            messagebox.showinfo("توجه", "هیچ فردی در تصویر شناسایی نشد")

        self.live_btn.config(
            state=tk.NORMAL if self.capture_running else tk.DISABLED)
        self._draw_boxes_and_show()
        self._set_status(
            f"{len(self.person_boxes)} فرد شناسایی شد — روی فرد مورد نظر کلیک کنید"
        )

    def _freeze_frame(self, frame):
        """نگه‌داشتن تصویر (دوربین یا فایل) و پاک کردن انتخاب قبلی."""
        self.frozen_frame = frame
        self.selected_box_idx = None
        self.current_embedding = None
        self.selected_face_crop = None
        if not self.pending_embeddings:
            self.send_btn.config(state=tk.DISABLED)
        self._clear_face_preview()

    def _detect_persons(self, frame):
        """اجرای YOLO روی تصویر و پر کردن person_boxes."""
        try:
            result = self.yolo_model.predict(
                frame, classes=[0], verbose=False)[0]
        except Exception as e:
            messagebox.showerror("خطا در تشخیص", str(e))
            return False

        self.person_boxes = []
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            self.person_boxes.append((x1, y1, x2, y2))
        return True

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
        canvas_w = self.canvas.winfo_width()
        canvas_h = self.canvas.winfo_height()
        if canvas_w < 10 or canvas_h < 10:
            canvas_w, canvas_h = 800, 600
        scale = min(canvas_w / w, canvas_h / h)
        new_w, new_h = int(w * scale), int(h * scale)
        resized = cv2.resize(rgb, (new_w, new_h))

        img = Image.fromarray(resized)
        # جلوگیری از garbage collection
        self._tk_img = ImageTk.PhotoImage(image=img)
        self.canvas.delete("all")
        self.canvas.create_image(
            canvas_w // 2, canvas_h // 2, image=self._tk_img)

        # برای تبدیل مختصات کلیک بعداً لازم داریم
        self._display_scale = scale
        self._display_offset = ((canvas_w - new_w) // 2,
                                (canvas_h - new_h) // 2)

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
            self._clear_face_preview()
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
        self._set_status(
            "چهره استخراج شد — می‌توانید آن را به لیست اضافه کنید")

    # -----------------------------------------------------------------
    # مدیریت لیست موقتِ چند عکس برای یک نفر
    # -----------------------------------------------------------------
    def add_to_pending(self):
        if self.current_embedding is None:
            return

        self.pending_embeddings.append(self.current_embedding)
        self.pending_crops.append(self.selected_face_crop)
        # آخرین چهره برای آپلود نگه داشته می‌شود
        self.pending_face_crop = self.selected_face_crop

        # آماده‌سازی برای گرفتن عکس بعدی از همان شخص —
        # فرم (نام/تولد/...) دست‌نخورده می‌ماند چون شخص همان است
        self.current_embedding = None
        self.selected_face_crop = None
        self.add_btn.config(state=tk.DISABLED)
        self._clear_face_preview()
        self.frozen_frame = None
        self.person_boxes = []
        self.selected_box_idx = None
        self.live_btn.config(state=tk.DISABLED)
        if not self.capture_running:
            self.canvas.delete("all")

        self._render_pending_thumbs()

        self._set_status(
            f"اضافه شد ({fa(len(self.pending_embeddings))} عکس در لیست) — "
            "برای عکس بعدی «گرفتن عکس» یا «عکس از فایل» را بزنید "
            "یا اطلاعات را تکمیل و ارسال کنید"
        )

    def _remove_pending(self, idx):
        """حذف یک عکس از لیست موقت (دکمه‌ی × روی بندانگشتی)."""
        if not 0 <= idx < len(self.pending_embeddings):
            return
        removed = idx + 1
        self.pending_embeddings.pop(idx)
        self.pending_crops.pop(idx)
        self.pending_face_crop = (
            self.pending_crops[-1] if self.pending_crops else None
        )
        self._render_pending_thumbs()
        if self.pending_embeddings:
            self._set_status(
                f"عکس شماره {fa(removed)} حذف شد "
                f"({fa(len(self.pending_embeddings))} عکس باقی مانده)"
            )
        else:
            self._set_status("همه عکس‌های اضافه‌شده حذف شد")

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
        self.pending_crops = []
        self.pending_face_crop = None
        self._render_pending_thumbs()

    # -----------------------------------------------------------------
    # نوار پیش‌نمایش بندانگشتیِ عکس‌های جمع‌شده
    # -----------------------------------------------------------------
    def _thumb_photo(self, bgr_crop, size=72):
        if bgr_crop is None or getattr(bgr_crop, "size", 0) == 0:
            img = Image.new("RGB", (size, size), (226, 232, 240))
        else:
            rgb = cv2.cvtColor(bgr_crop, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(rgb).resize((size, size))
        return ImageTk.PhotoImage(image=img)

    def _render_pending_thumbs(self):
        """بازسازی نوار پیش‌نمایش + به‌روزرسانی شمارنده و دکمه‌ها."""
        count = len(self.pending_embeddings)
        self.pending_label.config(text=f"عکس‌های اضافه‌شده: {fa(count)}")

        for child in self.thumbs_inner.winfo_children():
            child.destroy()
        self._thumb_imgs = []

        if count == 0:
            self.thumbs_bar.pack_forget()
            self.clear_pending_btn.config(state=tk.DISABLED)
            self.send_btn.config(state=tk.DISABLED)
            return

        self.thumbs_count.config(text=f"{fa(count)} عکس")
        self.clear_pending_btn.config(state=tk.NORMAL)
        self.send_btn.config(state=tk.NORMAL)

        for i, crop in enumerate(self.pending_crops):
            cell = tk.Frame(self.thumbs_inner, bg=C_SURFACE)
            # راست‌به‌چپ مثل بقیه‌ی رابط فارسی
            cell.pack(side=tk.RIGHT, padx=(14, 0), pady=4)

            photo = self._thumb_photo(crop)
            self._thumb_imgs.append(photo)
            thumb = tk.Label(
                cell, image=photo, bg=C_BG, cursor="hand2",
                highlightbackground=C_BORDER, highlightthickness=1,
            )
            thumb.pack()
            # کلیک روی بندانگشتی → نمایش بزرگ‌تر در کادر فرم
            thumb.bind("<Button-1>", lambda e, c=crop: self._show_face_preview(c))

            row = tk.Frame(cell, bg=C_SURFACE)
            row.pack(fill=tk.X, pady=(3, 0))
            tk.Label(
                row, text=f"#{fa(i + 1)}", bg=C_SURFACE, fg=C_MUTED,
                font=(self.font, 8),
            ).pack(side=tk.RIGHT)
            tk.Button(
                row, text="×", command=lambda idx=i: self._remove_pending(idx),
                bg="#fee2e2", fg="#dc2626", activebackground="#fecaca",
                activeforeground="#b91c1c", relief="flat", bd=0, padx=7,
                cursor="hand2", font=(self.font, 10, "bold"),
            ).pack(side=tk.LEFT)

        if not self.thumbs_bar.winfo_ismapped():
            self.thumbs_bar.pack(fill=tk.X, pady=(10, 0))

    def _on_thumbs_configure(self, event=None):
        """چسباندن نوار بندانگشتی به لبه‌ی راست + به‌روزرسانی ناحیه‌ی اسکرول."""
        width = self.thumbs_canvas.winfo_width()
        if width > 1:
            self.thumbs_canvas.coords(self._thumbs_win, width, 0)
        bbox = self.thumbs_canvas.bbox("all")
        if bbox:
            self.thumbs_canvas.configure(scrollregion=bbox)

    def _on_thumbs_wheel(self, event):
        """اسکرول افقی نوار بندانگشتی با اسکرول موس."""
        if not event.delta:
            return
        bbox = self.thumbs_canvas.bbox("all")
        if bbox and bbox[2] > self.thumbs_canvas.winfo_width():
            self.thumbs_canvas.xview_scroll(-1 * (event.delta // 120), "units")

    def _show_face_preview(self, bgr_face):
        if bgr_face is None or getattr(bgr_face, "size", 0) == 0:
            self._clear_face_preview()
            return
        rgb = cv2.cvtColor(bgr_face, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(rgb).resize((96, 96))
        self._tk_face_img = ImageTk.PhotoImage(image=img)
        self.face_preview_label.config(image=self._tk_face_img, text="")

    def _clear_face_preview(self):
        self.face_preview_label.config(image="", text="بدون چهره")

    # -----------------------------------------------------------------
    # تب «مدیریت افراد» — مشاهده، ویرایش و حذف رکوردهای ثبت‌شده
    # -----------------------------------------------------------------
    def _build_manage_tab(self):
        """چیدن تب دوم: نوار جستجو، جدول افراد و فرم ویرایش سمت راست."""
        wrap = tk.Frame(self.manage_tab, bg=C_BG)
        wrap.pack(fill=tk.BOTH, expand=True, padx=14, pady=14)

        # --- نوار ابزار بالای تب ---
        bar = tk.Frame(wrap, bg=C_BG)
        bar.pack(fill=tk.X, pady=(0, 10))

        tk.Label(bar, text="جستجو:", bg=C_BG, fg=C_MUTED,
                 font=(self.font, 9)).pack(side=tk.RIGHT)
        self.people_search_var = tk.StringVar()
        ttk.Entry(bar, textvariable=self.people_search_var,
                  width=30).pack(side=tk.RIGHT, padx=(6, 0))
        self.people_search_var.trace_add(
            "write", lambda *_: self._filter_people())

        self.people_count_label = tk.Label(
            bar, text="", bg=C_PRIMARY_SOFT, fg=C_PRIMARY,
            font=(self.font, 9, "bold"), padx=8,
        )
        self.people_count_label.pack(side=tk.RIGHT, padx=(0, 14))

        ttk.Button(bar, text="بروزرسانی فهرست", command=self.load_people,
                   style="Primary.TButton").pack(side=tk.LEFT)

        # --- کارت جدول (سمت چپ، قابل گسترش) ---
        list_card = tk.Frame(wrap, bg=C_SURFACE, highlightbackground=C_BORDER,
                             highlightthickness=1)
        list_card.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 12))

        # ترتیب ستون‌ها از چپ به راست است؛ چون رابط فارسی است «نام» باید
        # راست‌ترین ستون باشد، پس آخر از همه اضافه می‌شود.
        cols = ("userwhom", "socialnumber", "role", "gender", "age", "name")
        self.people_tree = ttk.Treeview(
            list_card, columns=cols, show="headings", selectmode="browse",
            style="People.Treeview",
        )
        headings = {
            "name": "نام", "age": "سن", "gender": "جنسیت",
            "role": "نقش", "socialnumber": "کد ملی", "userwhom": "userwhom",
        }
        widths = {
            "name": 170, "age": 60, "gender": 80,
            "role": 90, "socialnumber": 130, "userwhom": 120,
        }
        for c in cols:
            self.people_tree.heading(c, text=headings[c], anchor="e")
            self.people_tree.column(c, width=widths[c], anchor="e",
                                    stretch=(c == "name"))

        people_sb = ttk.Scrollbar(list_card, orient=tk.VERTICAL,
                                  command=self.people_tree.yview)
        self.people_tree.configure(yscrollcommand=people_sb.set)
        people_sb.pack(side=tk.LEFT, fill=tk.Y)
        self.people_tree.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        self.people_tree.bind("<<TreeviewSelect>>", self._on_people_select)

        # --- فرم ویرایش (سمت راست، عرض ثابت مثل فرم تب ثبت) ---
        panel = tk.Frame(wrap, bg=C_SURFACE, width=330,
                         highlightbackground=C_BORDER, highlightthickness=1)
        panel.pack(side=tk.RIGHT, fill=tk.Y)
        panel.pack_propagate(False)

        tk.Label(panel, text="ویرایش فرد", bg=C_SURFACE, fg=C_TEXT,
                 font=(self.font, 12, "bold")).pack(anchor="e", padx=16,
                                                     pady=(12, 0))
        ttk.Separator(panel, orient="horizontal").pack(
            fill=tk.X, padx=16, pady=(6, 8))

        photo_box = tk.Frame(panel, width=104, height=104, bg=C_BG,
                             highlightbackground=C_BORDER, highlightthickness=1)
        photo_box.pack(anchor="e", padx=16, pady=(2, 4))
        photo_box.pack_propagate(False)
        self.people_photo_label = tk.Label(
            photo_box, bg=C_BG, text="بدون تصویر", fg="#94a3b8",
            font=(self.font, 9),
        )
        self.people_photo_label.pack(expand=True, fill=tk.BOTH)

        self.p_name_var = tk.StringVar()
        self.p_age_var = tk.StringVar()
        self.p_gender_var = tk.StringVar()
        self.p_role_var = tk.StringVar()
        self.p_social_var = tk.StringVar()
        self.p_userwhom_var = tk.StringVar()

        fields = tk.Frame(panel, bg=C_SURFACE)
        fields.pack(fill=tk.X, padx=16, pady=(6, 0))

        def label_for(text):
            tk.Label(fields, text=text, bg=C_SURFACE, fg=C_MUTED,
                     font=(self.font, 9)).pack(anchor="e", pady=(7, 2))

        label_for("نام")
        ttk.Entry(fields, textvariable=self.p_name_var).pack(fill=tk.X)

        label_for("سن")
        ttk.Entry(fields, textvariable=self.p_age_var).pack(fill=tk.X)

        duo = tk.Frame(fields, bg=C_SURFACE)
        duo.pack(fill=tk.X)

        col_gender = tk.Frame(duo, bg=C_SURFACE)
        col_gender.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 4))
        tk.Label(col_gender, text="جنسیت", bg=C_SURFACE, fg=C_MUTED,
                 font=(self.font, 9)).pack(anchor="e", pady=(7, 2))
        ttk.Combobox(col_gender, textvariable=self.p_gender_var,
                     values=list(GENDER_MAP), state="readonly").pack(fill=tk.X)

        col_role = tk.Frame(duo, bg=C_SURFACE)
        col_role.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(4, 0))
        tk.Label(col_role, text="نقش", bg=C_SURFACE, fg=C_MUTED,
                 font=(self.font, 9)).pack(anchor="e", pady=(7, 2))
        ttk.Combobox(col_role, textvariable=self.p_role_var,
                     values=list(ROLE_MAP), state="readonly").pack(fill=tk.X)

        label_for("کد ملی")
        ttk.Entry(fields, textvariable=self.p_social_var).pack(
            fill=tk.X, pady=(0, 2))

        label_for("userwhom")
        ttk.Entry(fields, textvariable=self.p_userwhom_var).pack(
            fill=tk.X, pady=(0, 4))

        # --- دکمه‌ها (پایین فرم) ---
        actions = tk.Frame(panel, bg=C_SURFACE)
        actions.pack(side=tk.BOTTOM, fill=tk.X, padx=16, pady=12)

        self.p_save_btn = ttk.Button(
            actions, text="ذخیره تغییرات", command=self.save_person,
            state=tk.DISABLED, style="Success.TButton",
        )
        self.p_save_btn.pack(fill=tk.X, pady=(0, 6))

        self.p_delete_btn = ttk.Button(
            actions, text="حذف این فرد", command=self.delete_person,
            state=tk.DISABLED, style="Danger.TButton",
        )
        self.p_delete_btn.pack(fill=tk.X)

    def _on_tab_changed(self, event=None):
        """بارگذاری فهرست افراد در اولین بار که تب «مدیریت افراد» باز می‌شود."""
        if not self._people_loaded and self.notebook.index("current") == 1:
            self._people_loaded = True
            self.load_people()

    def load_people(self, event=None):
        """خواندن همه‌ی رکوردهای known_face از PocketBase و پر کردن جدول."""
        url = f"{POCKETBASE_URL}/api/collections/{COLLECTION}/records"
        records = []
        page = 1
        total_pages = 1
        try:
            while page <= total_pages:
                res = requests.get(
                    url,
                    params={"perPage": 200, "page": page, "sort": "name"},
                    timeout=REQUEST_TIMEOUT,
                )
                res.raise_for_status()
                data = res.json()
                records.extend(data.get("items", []))
                total_pages = data.get("totalPages", 1)
                page += 1
        except Exception as e:
            messagebox.showerror(
                "خطا", f"بارگذاری فهرست افراد ناموفق بود:\n{e}")
            return

        self.people_map = {
            rec["id"]: rec for rec in records if rec.get("id")
        }
        self._filter_people()
        self._set_status(
            f"{fa(len(self.people_map))} فرد از پایگاه داده خوانده شد")

    def _people_row(self, rec):
        """مقادیر نمایشی یک ردیف جدول (به ترتیب ستون‌ها)."""
        emb = rec.get("embdanings")
        photos = fa(len(emb) // EMBEDDING_DIM) if isinstance(emb, list) else "—"
        return (
            rec.get("userwhom") or "—",
            rec.get("socialnumber") or "—",
            ROLE_FA.get(rec.get("role"), "—"),
            GENDER_FA.get(rec.get("gender"), "—"),
            fa(rec.get("age") or "—"),
            rec.get("name") or "—",
        )

    def _filter_people(self):
        """بازسازی جدول با فیلتر جستجوی زنده (نام / کد ملی / userwhom)."""
        if not hasattr(self, "people_tree"):
            return

        query = self.people_search_var.get().translate(
            _PERSIAN_DIGITS).strip().lower()
        self.people_tree.delete(*self.people_tree.get_children())

        shown = 0
        for rid, rec in self.people_map.items():
            if query:
                haystack = " ".join(
                    str(rec.get(k) or "")
                    for k in ("name", "socialnumber", "userwhom")
                )
                if query not in haystack.translate(_PERSIAN_DIGITS).lower():
                    continue
            self.people_tree.insert(
                "", tk.END, iid=rid, values=self._people_row(rec))
            shown += 1

        self.people_count_label.config(text=f"{fa(shown)} نفر")
        if shown == 0:
            self._reset_people_form()
        else:
            self.p_save_btn.config(state=tk.DISABLED)
            self.p_delete_btn.config(state=tk.DISABLED)

    def _on_people_select(self, event=None):
        """پر کردن فرم ویرایش از روی ردیف انتخاب‌شده در جدول."""
        selection = self.people_tree.selection()
        if not selection:
            return
        rec = self.people_map.get(selection[0])
        if rec is None:
            return

        self.p_name_var.set(rec.get("name") or "")
        self.p_age_var.set(str(rec.get("age") or ""))
        self.p_gender_var.set(GENDER_FA.get(rec.get("gender"), ""))
        self.p_role_var.set(ROLE_FA.get(rec.get("role"), ""))
        self.p_social_var.set(rec.get("socialnumber") or "")
        self.p_userwhom_var.set(rec.get("userwhom") or "")

        self.p_save_btn.config(state=tk.NORMAL)
        self.p_delete_btn.config(state=tk.NORMAL)
        self._show_people_photo(rec)
        self._set_status(
            f"«{rec.get('name') or '—'}» انتخاب شد — فیلدها قابل ویرایش هستند")

    def _reset_people_form(self):
        """خالی کردن فرم ویرایش و غیرفعال کردن دکمه‌ها."""
        for var in (self.p_name_var, self.p_age_var, self.p_gender_var,
                    self.p_role_var, self.p_social_var, self.p_userwhom_var):
            var.set("")
        self.people_photo_label.config(image="", text="بدون تصویر")
        self._people_photo = None
        self.p_save_btn.config(state=tk.DISABLED)
        self.p_delete_btn.config(state=tk.DISABLED)

    def _show_people_photo(self, rec):
        """نمایش تصویر ذخیره‌شده‌ی رکورد (فیلد image) در کادر فرم."""
        filename = rec.get("image")
        if not filename:
            self.people_photo_label.config(image="", text="بدون تصویر")
            self._people_photo = None
            return

        url = (f"{POCKETBASE_URL}/api/files/{COLLECTION_ID}/"
               f"{rec['id']}/{filename}")
        try:
            res = requests.get(url, timeout=REQUEST_TIMEOUT)
            res.raise_for_status()
            img = Image.open(io.BytesIO(res.content)).convert("RGB")
            img = img.resize((96, 96), Image.LANCZOS)
        except Exception:
            self.people_photo_label.config(image="", text="تصویر موجود نیست")
            self._people_photo = None
            return

        self._people_photo = ImageTk.PhotoImage(image=img)
        self.people_photo_label.config(image=self._people_photo, text="")

    def save_person(self):
        """ذخیره‌ی فیلدهای ویرایش‌شده روی رکورد انتخاب‌شده (PATCH)."""
        selection = self.people_tree.selection()
        rec = self.people_map.get(selection[0]) if selection else None
        if rec is None:
            return

        name = self.p_name_var.get().strip()
        if not name:
            messagebox.showwarning("خطا", "وارد کردن نام الزامی است")
            return

        age_raw = self.p_age_var.get().strip().translate(_PERSIAN_DIGITS)
        if age_raw and not age_raw.isdigit():
            messagebox.showwarning("خطا", "سن باید یک عدد صحیح باشد")
            return

        # همه‌ی فیلدها متنی‌اند، پس مقدار خالی هم قابل ارسال است
        # (برخلاف append که نباید فیلد خالی را بازنویسی کند).
        data = {
            "name": name,
            "age": age_raw,
            "gender": GENDER_MAP.get(self.p_gender_var.get().strip(), ""),
            "role": ROLE_MAP.get(self.p_role_var.get().strip(), ""),
            "socialnumber": self.p_social_var.get().strip(),
            "userwhom": self.p_userwhom_var.get().strip(),
        }

        url = (f"{POCKETBASE_URL}/api/collections/{COLLECTION}"
               f"/records/{rec['id']}")
        try:
            res = requests.patch(url, data=data, timeout=REQUEST_TIMEOUT)
            res.raise_for_status()
        except Exception as e:
            messagebox.showerror("خطا در ذخیره", str(e))
            return

        # رکورد تازه‌خوانده‌شده را جایگزین می‌کنیم تا جدول و فرم هم‌گام بمانند
        try:
            fresh = requests.get(
                url, params={"fields": "id,name,age,gender,role,"
                                     "socialnumber,userwhom,image"},
                timeout=REQUEST_TIMEOUT,
            )
            fresh.raise_for_status()
            self.people_map[rec["id"]] = fresh.json()
        except Exception:
            self.people_map[rec["id"]].update(data)

        self._filter_people()
        # انتخاب را روی همان رکورد نگه می‌داریم تا ویرایش بعدی هم ممکن باشد
        if rec["id"] in self.people_tree.get_children():
            self.people_tree.selection_set(rec["id"])
            self._on_people_select()
        self._notify_engine_refresh()
        messagebox.showinfo("موفق", f"تغییرات «{name}» ذخیره شد")
        self._set_status(f"«{name}» به‌روزرسانی شد")

    def delete_person(self):
        """حذف کامل رکورد انتخاب‌شده از پایگاه داده (با تأییدیه)."""
        selection = self.people_tree.selection()
        rec = self.people_map.get(selection[0]) if selection else None
        if rec is None:
            return

        name = rec.get("name") or "—"
        if not messagebox.askyesno(
            "تأیید",
            f"رکورد «{name}» با تمام امبدینگ‌هایش برای همیشه حذف شود؟",
        ):
            return

        url = (f"{POCKETBASE_URL}/api/collections/{COLLECTION}"
               f"/records/{rec['id']}")
        try:
            res = requests.delete(url, timeout=REQUEST_TIMEOUT)
            res.raise_for_status()
        except Exception as e:
            messagebox.showerror("خطا در حذف", str(e))
            return

        self.people_map.pop(rec["id"], None)
        self._filter_people()
        self._notify_engine_refresh()
        messagebox.showinfo("موفق", f"«{name}» حذف شد")
        self._set_status(f"«{name}» از پایگاه داده حذف شد")

    def _notify_engine_refresh(self):
        """خبر کردن موتور تشخیص چهره که دیتابیس تغییر کرده است."""
        try:
            requests.get(f"http://127.0.0.1:{self.readPort()}/util/refreshDb",
                         timeout=5)
        except Exception:
            # نبودن موتور نباید عملیات را ناموفق جلوه کند
            pass

    # -----------------------------------------------------------------
    # ارسال به PocketBase
    # -----------------------------------------------------------------
    def send_to_db(self):
        if not self.pending_embeddings:
            messagebox.showwarning(
                "خطا", "حداقل یک چهره باید به لیست اضافه شده باشد")
            return

        name = self.name_var.get().strip()
        if not name:
            messagebox.showwarning("خطا", "وارد کردن نام الزامی است")
            return

        birth = parse_jalali_date(self.birth_var.get())
        if birth is None:
            messagebox.showwarning(
                "خطا", "تاریخ تولد شمسی معتبر وارد کنید (مثال: 1377/04/12)"
            )
            return
        age_value = jalali_age(birth)
        if age_value < 0:
            messagebox.showwarning("خطا", "تاریخ تولد نمی‌تواند در آینده باشد")
            return
        age = str(age_value)
        gender_fa = self.gender_var.get().strip()
        gender = GENDER_MAP.get(gender_fa, "")
        role_fa = self.role_var.get().strip()
        role = ROLE_MAP.get(role_fa, "")
        social = self.social_var.get().strip()
        userwhom_fa = self.userwhom_var.get().strip()
        userwhom_map={"همکار":"colleague","ارباب رجوع":"visitor"}
        userwhom=userwhom_map.get(userwhom_fa,"")


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
                self._append_embedding(
                    existing, embedding_list, age, gender, role, social,userwhom)
                messagebox.showinfo(
                    "موفق", f"{count} امبدینگ جدید به «{name}» اضافه شد")
            else:
                self._create_record(name, embedding_list,
                                    age, gender, role, social,userwhom)
                messagebox.showinfo("موفق", f"«{name}» با {count} عکس ثبت شد")

            self._notify_engine_refresh()
            self._set_status("ارسال با موفقیت انجام شد")

            # ریست کامل فرم: عکس‌های لیست، تصویر freezes و همه‌ی فیلدها
            self.back_to_live()
            self._reset_pending()
            self._reset_enroll_form()

        except Exception as e:
            messagebox.showerror("خطا در ارسال", str(e))

    def _reset_enroll_form(self):
        """صفر کردن کامل فرم ثبت: همه‌ی فیلدها، پیش‌نمایش و دکمه‌ها.

        مقادیر از طریق StringVar پاک می‌شوند تا کمبوهای readonly هم
        واقعاً خالی دیده شوند."""
        for var in (self.name_var, self.birth_var, self.gender_var,
                    self.role_var, self.social_var, self.userwhom_var):
            var.set("")
        self.age_label.config(text="سن: —")
        self._clear_face_preview()
        self.add_btn.config(state=tk.DISABLED)
        self.live_btn.config(state=tk.DISABLED)
        self.current_embedding = None
        self.selected_face_crop = None
        self._tk_face_img = None
        # فوکوس از فیلدها برداشته می‌شود تا تایپ تصادفی وارد فرم نشود
        self.root.focus_set()

    def readPort(self):
        with open('hostname.json') as file:

            data = json.load(file)
            return data['port']

    def _find_existing_record(self, name):
        url = f"{POCKETBASE_URL}/api/collections/{COLLECTION}/records"
        # اسم را داخل فیلتر escape می‌کنیم
        safe_name = name.replace('"', '\\"')
        params = {"filter": f'name="{safe_name}"', "perPage": 1}
        res = requests.get(url, params=params, timeout=REQUEST_TIMEOUT)
        res.raise_for_status()
        items = res.json().get("items", [])
        return items[0] if items else None

    def _create_record(self, name, embedding_list, age, gender, role, social,userwhom):
        url = f"{POCKETBASE_URL}/api/collections/{COLLECTION}/records"
        data = {
            "name": name,
            "embdanings": json.dumps(embedding_list),
            "age": age,
            "gender": gender,
            "role": role,
            "socialnumber": social,
            "userwhom":userwhom
        }
        files = self._face_file_payload()
        res = requests.post(url, data=data, files=files,
                            timeout=REQUEST_TIMEOUT)
        res.raise_for_status()

    def _append_embedding(self, existing_record, new_embedding_list, age, gender, role, social,userwhom):
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
        if userwhom:
            data['userwhom']=userwhom

        files = self._face_file_payload()
        res = requests.patch(url, data=data, files=files,
                             timeout=REQUEST_TIMEOUT)
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
