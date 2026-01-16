# =============================================================
#  Optimized Abaca Grade GUI (keeps design, improved responsiveness)
# =============================================================
import os
import sys
import time
import threading
import datetime
import warnings
import json

import customtkinter as ctk
from tkinter import messagebox, filedialog
from PIL import Image, ImageTk
import cv2
import numpy as np
import joblib

from skimage.feature import local_binary_pattern, graycomatrix, graycoprops

# ---------------- Utility: resource path ----------------
def resource_path(relative_path):
    """Get absolute path to resource (for both dev and PyInstaller)."""
    try:
        base_path = sys._MEIPASS  # When bundled by PyInstaller
    except Exception:
        base_path = os.path.abspath(".")
    return os.path.join(base_path, relative_path)


# ---------------- Load model & preprocessors (unchanged) ----------------
MODEL_PATH  = resource_path("abaca_svm_model_pca.pkl")
SCALER_PATH = resource_path("scaler.pkl")
PCA_PATH    = resource_path("pca.pkl")
CLASS_PATH  = resource_path("classes.json")

for path in [MODEL_PATH, SCALER_PATH, PCA_PATH, CLASS_PATH]:
    if not os.path.exists(path):
        # Use messagebox only when root exists; print first then try messagebox
        print(f"Missing required file: {os.path.basename(path)}")
        try:
            from tkinter import Tk
            _tmp = Tk(); _tmp.withdraw()
            messagebox.showerror("Error", f"Missing required file: {os.path.basename(path)}")
            _tmp.destroy()
        except Exception:
            pass
        raise SystemExit

print("✅ Loading model and preprocessors...")
model  = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)
pca    = joblib.load(PCA_PATH)
with open(CLASS_PATH) as f:
    CLASSES = json.load(f)
print("✅ Model and PCA pipeline ready.")

# ---------------- Texture + CNN features (unchanged logic) ----------------
radius = 3
n_points = 8 * radius
cnn_model = None  # lazy loaded

def extract_texture_features(img_gray):
    img_gray = img_gray.astype(np.uint8)
    lbp = local_binary_pattern(img_gray, n_points, radius, method="uniform")
    hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, n_points + 3),
                           range=(0, n_points + 2))
    hist = hist.astype("float")
    hist /= (hist.sum() + 1e-6)
    glcm = graycomatrix(img_gray, [1], [0], 256, symmetric=True, normed=True)
    contrast = graycoprops(glcm, "contrast")[0, 0]
    homogeneity = graycoprops(glcm, "homogeneity")[0, 0]
    energy = graycoprops(glcm, "energy")[0, 0]
    return np.hstack([hist, contrast, homogeneity, energy])

def extract_cnn_features(img):
    """CNN embeddings using EfficientNetB0. Requires cnn_model to be loaded."""
    img_resized = cv2.resize(img, (224, 224))
    img_batch = np.expand_dims(img_resized.astype("float32"), axis=0)
    feats = cnn_model.predict(img_batch, verbose=0).flatten()
    return feats

def preprocess_image(img):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    texture_feat = extract_texture_features(gray)
    cnn_feat = extract_cnn_features(img)
    combined = np.hstack([texture_feat, cnn_feat]).reshape(1, -1)
    scaled = scaler.transform(combined)
    reduced = pca.transform(scaled)
    return reduced

def classify_image_svm(img):
    processed = preprocess_image(img)
    probs = model.predict_proba(processed)[0]
    idx = int(np.argmax(probs))
    label = CLASSES[idx] if isinstance(CLASSES, list) else CLASSES[str(idx)]
    conf = probs[idx] * 100
    return label, conf

# ---------------- Class info mapping (kept) ----------------
CLASS_INFO = {
    "H": {"name": "Soft Brown", "desc": "Dark brown."},
    "JK": {"name": "Seconds", "desc": "Dull brown to dingy light brown or dingy light yellow, frequently streak with light green."},
    "I": {"name": "Current", "desc": "Very light brown to light brown."},
    "S2": {"name": "Streaky Two", "desc": "Ivory white, slightly  tinged with very light brown to red or purple streak"},
    "NOT ABACA": {"name": "Not Abaca", "desc": "The sample is not classified as abaca."},
}

warnings.filterwarnings("ignore", category=UserWarning, module="customtkinter")

# ---------------- customtkinter base config ----------------
ctk.set_appearance_mode("light")
ctk.set_default_color_theme("blue")
ctk.set_widget_scaling(0.95)   # small speed boost on low-power devices
ctk.set_window_scaling(0.95)

# ---------------- Root window ----------------
DESIGN_MODE = True   # keep simulation as before
root = ctk.CTk()
root.title("ABACAGRADE")
if DESIGN_MODE:
    root.geometry("800x480")
    root.resizable(False, False)
else:
    screen_w = root.winfo_screenwidth()
    screen_h = root.winfo_screenheight()
    if screen_w <= 800 and screen_h <= 480:
        root.geometry("800x480")
        root.resizable(False, False)
    else:
        try:
            root.state("zoomed")
        except Exception:
            root.attributes("-fullscreen", True)

# allow ESC to toggle fullscreen (keep)
def toggle_fullscreen(event=None):
    fs = root.attributes("-fullscreen")
    root.attributes("-fullscreen", not fs)
root.bind("<Escape>", toggle_fullscreen)


# ---------------- Global image cache (load once) ----------------
# Use BILINEAR resizing when making thumbnails/previews (much faster than LANCZOS)
def _pil_open(path):
    return Image.open(resource_path(path)).convert("RGBA")

IMAGE_CACHE = {}
def cache_image(key, path, size=None):
    if key in IMAGE_CACHE:
        return IMAGE_CACHE[key]
    img = _pil_open(path)
    if size:
        img = img.resize(size, Image.BILINEAR)
    im = ImageTk.PhotoImage(img)
    IMAGE_CACHE[key] = im
    return im

# Pre-cache icons and bg (only once)
try:
    ICON_BACK   = cache_image("back", "images/back.png", (24, 24))
    ICON_CAMERA = cache_image("camera_small", "images/camera.png", (90, 90))
    ICON_UPLOAD = cache_image("upload_small", "images/upload.png", (90, 90)) 
    BG_PIL      = _pil_open("images/1-BG.png")   # keep PIL Image for fast resize on show
    LOGO_IMG    = cache_image("logo", "images/logoo.png", (120, 40))
except Exception as e:
    print("Warning: some images missing or could not be opened:", e)
    ICON_BACK = ICON_CAMERA = ICON_UPLOAD = LOGO_IMG = None
    BG_PIL = None

# ---------------- Single-page state manager ----------------
main_container = ctk.CTkFrame(root, fg_color="transparent")
main_container.pack(fill="both", expand=True)

_current_page_builder = None
_active_widgets = []
_previous_page_builder = None


def clear_active_widgets():
    global _active_widgets
    for w in _active_widgets:
        try:
            w.destroy()
        except Exception:
            pass
    _active_widgets = []

def show_page(builder, remember_previous=False):
    global _current_page_builder, _active_widgets, _previous_page_builder
    
    stop_camera()
    
    if remember_previous:
        _previous_page_builder = _current_page_builder

    clear_active_widgets()
    new_widgets = builder(main_container)
    _active_widgets = new_widgets
    _current_page_builder = builder
    
def go_back_to_previous_page():
    global _previous_page_builder
    if _previous_page_builder:
        show_page(_previous_page_builder)
    else:
        show_page(build_start_page)



# ---------------- Lightweight loading overlay ----------------
loading_overlay = None
loading_shown_time = None
loading_lock = threading.Lock()

def show_loading(message="Processing..."):
    global loading_overlay, loading_shown_time
    with loading_lock:
        if loading_overlay is not None:
            # update message if already visible
            try:
                loading_overlay.label.configure(text=message)
            except Exception:
                pass
            return

        # --- Create modal ---
        top = ctk.CTkToplevel(root)
        top.overrideredirect(True)
        top.attributes("-topmost", True)
        top.configure(fg_color="white")

        # --- Center positioning ---
        w, h = 320, 130
        root_x = root.winfo_rootx()
        root_y = root.winfo_rooty()
        root_w = root.winfo_width()
        root_h = root.winfo_height()
        x = root_x + (root_w // 2 - w // 2)
        y = root_y + (root_h // 2 - h // 2)
        top.geometry(f"{w}x{h}+{x}+{y}")

        # --- Inner frame ---
        frame = ctk.CTkFrame(
            top,
            fg_color="white",
            corner_radius=0,
            border_width=0,
            border_color="#f0f0f0",
            width=w,
            height=h
        )
        frame.pack(expand=True, fill="both")

        # --- Label (same style as original loader) ---
        lbl = ctk.CTkLabel(
            frame,
            text=message,
            font=("Poppins", 18, "bold"),
            text_color="#1E1E1E"
        )
        lbl.pack(pady=(35, 10))

        # --- Animated gold progress bar (same theme) ---
        bar = ctk.CTkProgressBar(
            frame,
            width=200,
            height=10,
            corner_radius=100,
            fg_color="#f5f5f5",
            progress_color="#d19c26",
            mode="indeterminate"
        )
        bar.pack()
        bar.start()

        # --- Attach references ---
        top.label = lbl
        top.bar = bar

        loading_overlay = top
        loading_shown_time = datetime.datetime.now()


def hide_loading(min_visible_time=0.3):
    global loading_overlay, loading_shown_time
    with loading_lock:
        if loading_overlay is None:
            return

        elapsed = (datetime.datetime.now() - loading_shown_time).total_seconds()
        remaining = max(0, min_visible_time - elapsed)

        def _destroy():
            global loading_overlay
            try:
                loading_overlay.destroy()
            except Exception:
                pass
            loading_overlay = None

        root.after(int(remaining * 1000), _destroy)

#Press animation
def add_press_effect(box, callback, parent):
    orig = box.cget("fg_color")
    press_color = "#b8861a"  # darker shade of your accent color

    # When finger/mouse is PRESSED DOWN
    def on_press(event):
        try:
            box.configure(fg_color=press_color)
        except:
            pass

    # When finger/mouse RELEASES
    def on_release(event):
        try:
            box.configure(fg_color=orig)
        except:
            pass
        parent.after(120, callback)   # delay so animation is visible

    box.bind("<ButtonPress-1>", on_press)
    box.bind("<ButtonRelease-1>", on_release)



# ---------------- Page builders (maintain original design) ----------------
# We'll implement each page as a function that returns created widgets (so they can be destroyed fast).
# Page 1: Start Screen
def build_start_page(parent):
    widgets = []

    def update_bg(event=None):
        if BG_PIL is not None:
            w = parent.winfo_width()
            h = parent.winfo_height()

            bg = BG_PIL.resize((w, h), Image.BILINEAR)
            bg_imgtk = ImageTk.PhotoImage(bg)
            bg_label.configure(image=bg_imgtk)
            bg_label.image = bg_imgtk

    # background
    if BG_PIL is not None:
        bg_label = ctk.CTkLabel(parent, text="")
        bg_label.place(x=0, y=0, relwidth=1, relheight=1)
        widgets.append(bg_label)

        # update background when window resizes
        parent.bind("<Configure>", update_bg)
        parent.after(100, update_bg)
    else:
        bg_label = ctk.CTkLabel(parent, text="")
        bg_label.place(x=0, y=0, relwidth=1, relheight=1)
        widgets.append(bg_label)

    start_btn = ctk.CTkButton(
        parent,
        text="START CLASSIFICATION",
        font=("Segoe UI", 18, "bold"),
        fg_color="#E0E0E0",
        text_color="black",
        hover_color="#353A40",
        corner_radius=0,
        width=260,
        height=50,
        command=lambda: show_page(build_select_page)
    )
    start_btn.place(relx=0.5, rely=0.78, anchor="center")
    widgets.append(start_btn)
    
    return widgets

#Showing the Info MEnu in the starting page
# Showing the Info Menu (UPDATED: only 2 buttons, Page 2 style layout)
def build_info_menu_page(parent):
    widgets = []
    poppins_bold = ("Poppins", 28, "bold")
    poppins_medium = ("Poppins", 18, "bold")
    accent_color = "#d19c26"
    text_dark = "#1E1E1E"

    # Back Button
    back_btn = ctk.CTkButton(
        parent, text="", image=ICON_BACK,
        fg_color=accent_color, hover_color="#c18b20",
        corner_radius=50, width=40, height=40,
        command=lambda: go_back_to_previous_page()
    )
    back_btn.place(x=20, y=20)
    widgets.append(back_btn)

    # Title
    title_lbl = ctk.CTkLabel(
        parent,
        text="INFORMATION MENU",
        font=poppins_bold,
        text_color=text_dark
    )
    title_lbl.place(relx=0.5, y=70, anchor="center")
    widgets.append(title_lbl)

    # ---- Box Size ----
    BOX_W = 220
    BOX_H = 200
    RADIUS = 30

    # ======================
    #   LEFT BOX – ABACA GRADES
    # ======================
    box1 = ctk.CTkFrame(
        parent, width=BOX_W, height=BOX_H,
        fg_color=accent_color, corner_radius=RADIUS
    )
    box1.place(relx=0.30, rely=0.55, anchor="center")
    box1.pack_propagate(False)
    widgets.append(box1)

    lbl1 = ctk.CTkLabel(
        box1, text="Abaca Grades",
        font=poppins_medium, text_color="white"
    )
    lbl1.pack(expand=True)
    widgets.append(lbl1)

    add_press_effect(box1, lambda: show_page(build_info_page), parent)

    # ======================
    #   RIGHT BOX – HOW TO USE (USER MANUAL)
    # ======================
    box2 = ctk.CTkFrame(
        parent, width=BOX_W, height=BOX_H,
        fg_color=accent_color, corner_radius=RADIUS
    )
    box2.place(relx=0.70, rely=0.55, anchor="center")
    box2.pack_propagate(False)
    widgets.append(box2)

    lbl2 = ctk.CTkLabel(
        box2, text="How to Use",
        font=poppins_medium, text_color="white"
    )
    lbl2.pack(expand=True)
    widgets.append(lbl2)

    add_press_effect(box2, lambda: show_page(build_guide_page), parent)

    return widgets




#Showing the INFO of abaca Grades fiber
def build_info_page(parent):
    widgets = []
    poppins_bold = ("Poppins", 26, "bold")
    poppins_medium = ("Poppins", 16, "bold")
    open_sans = ("Open Sans", 13)
    accent_color = "#d19c26"
    text_dark = "#1E1E1E"
    gray_text = "#777777"

    # --- Back Button ---
    back_btn = ctk.CTkButton(
        parent, text="", image=ICON_BACK,
        fg_color=accent_color, hover_color="#c18b20",
        corner_radius=50, width=40, height=40,
        command=lambda: show_page(build_info_menu_page)
    )
    back_btn.place(x=20, y=20)
    widgets.append(back_btn)

    # --- Title ---
    title_lbl = ctk.CTkLabel(
        parent,
        text="ABACA GRADE INFORMATION TABLE",
        font=poppins_bold,
        text_color=text_dark
    )
    title_lbl.place(relx=0.5, y=60, anchor="center")
    widgets.append(title_lbl)

    # --- Table Container ---
    table_frame = ctk.CTkFrame(
        parent,
        fg_color="white",
        corner_radius=12,
        border_width=1,
        border_color="#e6e6e6"
    )
    table_frame.place(relx=0.5, rely=0.56, anchor="center", relwidth=0.90, relheight=0.72)
    widgets.append(table_frame)

    # --- Table Headers ---
    headers = ["Name", "Grade", "Extraction Source", "Fiber Strand (mm)", "Color", "Stripping", "Texture"]
    header_x = [10, 150, 240, 430, 560, 720, 820]

    for i, h in enumerate(headers):
        lbl = ctk.CTkLabel(
            table_frame,
            text=h,
            font=poppins_medium,
            text_color=text_dark
        )
        lbl.place(x=header_x[i], y=10)
        widgets.append(lbl)

    # --- Table Data ---
    table_data = [
        ["Streaky Two", "S2", "Next to the outer leafsheath", "0.20–0.50",
         "Ivory white, slight brown/red streaks", "Excellent", "Soft"],

        ["Current", "I", "Inner and middle leafsheath", "0.51–0.99",
         "Very light brown to light brown", "Good", "Medium soft"],

        ["Soft Brown", "H", "Outer leafsheath", "0.51–0.99",
         "Dark brown", "Good", ""],

        ["Seconds", "JK", "Inner, middle + next to outer leafsheath", "1.00–1.50",
         "Dull to dingy brown/yellow, streaked", "Fair", ""]
    ]

    row_y = 50
    row_gap = 70

    for row in table_data:
        for i, value in enumerate(row):
            lbl = ctk.CTkLabel(
                table_frame,
                text=value,
                font=open_sans,
                text_color=gray_text,
                justify="left",
                anchor="w",
                wraplength=160
            )
            lbl.place(x=header_x[i], y=row_y)
            widgets.append(lbl)
        row_y += row_gap

    return widgets

#Shwowing the Guide on how to use the app
def build_guide_page(parent):
    widgets = []
    accent = "#d19c26"
    text_dark = "#1E1E1E"

    back_btn = ctk.CTkButton(
        parent, text="", image=ICON_BACK,
        fg_color=accent, hover_color="#c18b20",
        corner_radius=50, width=40, height=40,
        command=lambda: show_page(build_info_menu_page)
    )
    back_btn.place(x=20, y=20)
    widgets.append(back_btn)

    title = ctk.CTkLabel(
        parent,
        text="HOW TO USE THE APP",
        font=("Poppins", 28, "bold"),
        text_color=text_dark
    )
    title.place(relx=0.5, y=70, anchor="center")
    widgets.append(title)

    guide_text = (
        "1. Select a mode: Live Camera or Upload Image.\n\n"
        "2. Capture or upload an image of an Abaca fiber bundle.\n\n"
        "3. Wait for the prediction results.\n\n"
        "4. View the confidence levels and description.\n\n"
        "5. Press Classify Again or upload another image."
    )

    lbl = ctk.CTkLabel(
        parent,
        text=guide_text,
        font=("Open Sans", 14),
        text_color="#555555",
        justify="left",
        wraplength=700
    )
    lbl.place(relx=0.5, rely=0.55, anchor="center")
    widgets.append(lbl)

    return widgets




# Page 2: Mode Selection
def safe_configure(widget, **kwargs):
    try:
        widget.configure(**kwargs)
    except Exception:
        pass

def build_select_page(parent):
    widgets = []
    poppins_bold = ("Poppins", 28, "bold")
    poppins_medium = ("Poppins", 18, "bold")
    accent_color = "#d19c26"

    # Back button
    back_btn_sel = ctk.CTkButton(
        parent, text="", image=ICON_BACK,
        fg_color=accent_color, hover_color="#c18b20",
        corner_radius=50, width=40, height=40,
        command=lambda: show_page(build_start_page)
    )
    back_btn_sel.place(x=20, y=20)
    widgets.append(back_btn_sel)

    # Title
    select_lbl = ctk.CTkLabel(
        parent, text="CHOOSE MODE",
        font=poppins_bold, text_color="#1E1E1E"
    )
    select_lbl.place(relx=0.5, y=60, anchor="center")
    widgets.append(select_lbl)

    # ======================
    #   CAMERA BOX
    # ======================
    camera_box = ctk.CTkFrame(
        parent, width=220, height=200,
        fg_color=accent_color, corner_radius=30
    )
    camera_box.place(relx=0.3, rely=0.55, anchor="center")
    camera_box.pack_propagate(False)
    widgets.append(camera_box)

    cam_icon_label = ctk.CTkLabel(camera_box, image=ICON_CAMERA, text="")
    cam_icon_label.pack(pady=(30, 10))
    widgets.append(cam_icon_label)

    cam_text_label = ctk.CTkLabel(
        camera_box, text="Live Camera",
        font=poppins_medium, text_color="white"
    )
    cam_text_label.pack()
    widgets.append(cam_text_label)

    # OLD TAP EFFECT simple flash then go to page
    def on_camera_tap(e=None):
        original = camera_box.cget("fg_color")
        camera_box.configure(fg_color="#C7C7C7")  # flash effect
        parent.after(120, lambda: safe_configure(camera_box, fg_color=original))
        show_page(build_camera_page)

    for widget_click in (camera_box, cam_icon_label, cam_text_label):
        widget_click.bind("<Button-1>", on_camera_tap)

    # ======================
    #   UPLOAD BOX
    # ======================
    upload_box = ctk.CTkFrame(
        parent, width=220, height=200,
        fg_color=accent_color, corner_radius=30
    )
    upload_box.place(relx=0.7, rely=0.55, anchor="center")
    upload_box.pack_propagate(False)
    widgets.append(upload_box)

    upload_icon_label = ctk.CTkLabel(upload_box, image=ICON_UPLOAD, text="")
    upload_icon_label.pack(pady=(30, 10))
    widgets.append(upload_icon_label)

    upload_text_label = ctk.CTkLabel(
        upload_box,
        text="Upload Image",
        font=poppins_medium,
        text_color="white"
    )
    upload_text_label.pack()
    widgets.append(upload_text_label)

    # OLD TAP EFFECT
    def on_upload_tap(e=None):
        original = upload_box.cget("fg_color")
        upload_box.configure(fg_color="#C7C7C7")
        parent.after(120, lambda: safe_configure(upload_box, fg_color=original))
        show_page(build_upload_page)

    for widget_click in (upload_box, upload_icon_label, upload_text_label):
        widget_click.bind("<Button-1>", on_upload_tap)

    return widgets



# Page 3: Live Camera Page (preserve layout and elements)
# We'll create only the widgets needed and start camera loop separately
# Camera variables (global)
cap = None
camera_running = False
first_camera_frame = False

# Variables used by classify functions need to exist (we'll create labels in builder and keep references)
_camera_refs = {}

def build_camera_page(parent):
    widgets = []
    poppins_bold = ("Poppins", 28, "bold")
    poppins_medium = ("Poppins", 20, "bold")
    open_sans = ("Open Sans", 14)
    accent_color = "#d19c26"
    text_dark = "#1E1E1E"
    gray_text = "#777777"

    title_lbl_camera = ctk.CTkLabel(parent, text="LIVE CAMERA SELECTION", font=poppins_bold, text_color=text_dark)
    title_lbl_camera.place(relx=0.5, rely=0.08, anchor="center")
    widgets.append(title_lbl_camera)

    def back_from_camera():
        stop_camera()
        show_page(build_select_page)

        # If camera running, stop it then go back (with small loader)

        def task():
            stop_camera()
            root.after(0, lambda: show_page(build_select_page))

        threading.Thread(target=task, daemon=True).start()

    back_btn_camera = ctk.CTkButton(parent, text="", image=ICON_BACK, fg_color=accent_color,
                                   hover_color="#c18b20", corner_radius=50, width=40, height=40,
                                   command=back_from_camera)
    back_btn_camera.place(x=20, y=20)
    widgets.append(back_btn_camera)

    # camera preview frame
    camera_box = ctk.CTkFrame(parent, fg_color="white", corner_radius=15, border_width=1, border_color="#e5e5e5")
    camera_box.place(relx=0.30, rely=0.49, anchor="center", relwidth=0.50, relheight=0.55)
    camera_box.pack_propagate(False)
    widgets.append(camera_box)

    camera_lbl = ctk.CTkLabel(camera_box, text="Camera not started", width=480, height=280, text_color="gray")
    camera_lbl.pack(expand=True)
    widgets.append(camera_lbl)

    # left text outputs
    main_result_lbl_camera = ctk.CTkLabel(parent, text="Waiting for prediction...", font=poppins_medium, text_color=text_dark, anchor="w", justify="left")
    main_result_lbl_camera.place(relx=0.048, rely=0.770)
    widgets.append(main_result_lbl_camera)

    underline_frame_camera = ctk.CTkFrame(parent, fg_color=accent_color, height=9, width=160, corner_radius=5)
    underline_frame_camera.place(relx=0.048, rely=0.830)
    widgets.append(underline_frame_camera)

    desc_lbl_camera = ctk.CTkLabel(parent, text="", font=open_sans, text_color=gray_text, wraplength=420, justify="left", anchor="w")
    desc_lbl_camera.place(relx=0.048, rely=0.855)
    widgets.append(desc_lbl_camera)

    # Right side progress bars & labels
    progress_bars_camera = []
    progress_labels_camera = []
    base_x = 0.60; base_y = 0.25; gap_y = 0.1
    for i in range(4):
        lbl = ctk.CTkLabel(parent, text=f"Class {i+1}: (0%)", font=("Poppins", 16, "bold") if i == 0 else ("Poppins", 15), text_color=text_dark if i == 0 else gray_text)
        lbl.place(relx=base_x, rely=base_y + (i * gap_y), anchor="w")
        widgets.append(lbl); progress_labels_camera.append(lbl)

        bar = ctk.CTkProgressBar(parent, width=280, height=8, corner_radius=100, progress_color=accent_color if i == 0 else "#d1d1d1", fg_color="#f0f0f0")
        bar.place(relx=base_x, rely=base_y + 0.035 + (i * gap_y), anchor="w")
        bar.set(0)
        widgets.append(bar); progress_bars_camera.append(bar)

    # classify button
    classify_btn = ctk.CTkButton(parent, text="Classify", font=("Poppins", 16, "bold"), fg_color=accent_color, hover_color="#c18b20", text_color="white", corner_radius=25, width=150, height=42, command=lambda: classify_camera_frame())
    classify_btn.place(relx=0.85, rely=0.70, anchor="center")
    widgets.append(classify_btn)

    # bottom-right logo (optional)
    if LOGO_IMG:
        logo_lbl = ctk.CTkLabel(parent, image=LOGO_IMG, text="")
        logo_lbl.place(relx=0.97, rely=0.95, anchor="se")
        logo_lbl.bind("<Button-1>", lambda e: show_page(build_info_menu_page, remember_previous=True))
        widgets.append(logo_lbl)

    # store references needed by camera/classify functions
    _camera_refs['camera_box'] = camera_box
    _camera_refs['camera_lbl'] = camera_lbl
    _camera_refs['main_result_lbl_camera'] = main_result_lbl_camera
    _camera_refs['desc_lbl_camera'] = desc_lbl_camera
    _camera_refs['progress_bars_camera'] = progress_bars_camera
    _camera_refs['progress_labels_camera'] = progress_labels_camera
    _camera_refs['classify_btn'] = classify_btn

    # start camera feed (non-blocking)
    start_camera()
    return widgets

# Page 4: Upload Page
_upload_refs = {}
def build_upload_page(parent):
    widgets = []
    poppins_bold = ("Poppins", 28, "bold")
    poppins_medium = ("Poppins", 20, "bold")
    open_sans = ("Open Sans", 14)
    accent_color = "#d19c26"
    text_dark = "#1E1E1E"
    gray_text = "#777777"

    title_lbl_upload = ctk.CTkLabel(parent, text="UPLOAD IMAGE SECTION", font=poppins_bold, text_color=text_dark)
    title_lbl_upload.place(relx=0.5, rely=0.08, anchor="center")
    widgets.append(title_lbl_upload)

    back_btn_upload = ctk.CTkButton(parent, text="", image=ICON_BACK, fg_color=accent_color, hover_color="#c18b20", corner_radius=50, width=40, height=40, command=back_from_upload)
    back_btn_upload.place(x=20, y=20)
    widgets.append(back_btn_upload)

    upload_box = ctk.CTkFrame(parent, fg_color="white", corner_radius=15, border_width=1, border_color="#e5e5e5")
    upload_box.place(relx=0.30, rely=0.49, anchor="center", relwidth=0.50, relheight=0.55)
    upload_box.pack_propagate(False)
    widgets.append(upload_box)

    upload_lbl = ctk.CTkLabel(upload_box, text="No image uploaded", width=480, height=280, text_color="gray")
    upload_lbl.pack(expand=True)
    widgets.append(upload_lbl)

    main_result_lbl = ctk.CTkLabel(parent, text="Waiting for prediction...", font=("Poppins", 20, "bold"), text_color=text_dark, anchor="w", justify="left")
    main_result_lbl.place(relx=0.048, rely=0.770)
    widgets.append(main_result_lbl)

    underline_frame = ctk.CTkFrame(parent, fg_color=accent_color, height=9, width=160, corner_radius=5)
    underline_frame.place(relx=0.048, rely=0.830)
    widgets.append(underline_frame)

    desc_lbl_upload = ctk.CTkLabel(parent, text="", font=open_sans, text_color=gray_text, wraplength=420, justify="left", anchor="w")
    desc_lbl_upload.place(relx=0.048, rely=0.855)
    widgets.append(desc_lbl_upload)

    progress_bars = []
    progress_labels = []
    base_x = 0.60; base_y = 0.25; gap_y = 0.1
    for i in range(4):
        lbl = ctk.CTkLabel(parent, text=f"Class {i+1}: (0%)", font=("Poppins", 16, "bold") if i == 0 else ("Poppins", 15), text_color=text_dark if i == 0 else gray_text)
        lbl.place(relx=base_x, rely=base_y + (i * gap_y), anchor="w")
        widgets.append(lbl); progress_labels.append(lbl)

        bar = ctk.CTkProgressBar(parent, width=280, height=8, corner_radius=100, progress_color=accent_color if i == 0 else "#d1d1d1", fg_color="#f0f0f0")
        bar.place(relx=base_x, rely=base_y + 0.035 + (i * gap_y), anchor="w")
        bar.set(0)
        widgets.append(bar); progress_bars.append(bar)

    upload_btn = ctk.CTkButton(parent, text="Upload Image", font=("Poppins", 16, "bold"), fg_color=accent_color, hover_color="#c18b20", text_color="white", corner_radius=25, width=150, height=42, command=lambda: open_and_classify_image(upload_lbl, main_result_lbl, desc_lbl_upload, progress_bars, progress_labels, upload_btn))
    upload_btn.place(relx=0.80, rely=0.70, anchor="center")
    widgets.append(upload_btn)

    if LOGO_IMG:
        logo_lbl = ctk.CTkLabel(parent, image=LOGO_IMG, text="")
        logo_lbl.place(relx=0.97, rely=0.95, anchor="se")
        logo_lbl.bind("<Button-1>", lambda e: show_page(build_info_menu_page, remember_previous=True))
        widgets.append(logo_lbl)

    _upload_refs['upload_box'] = upload_box
    _upload_refs['upload_lbl'] = upload_lbl
    _upload_refs['main_result_lbl'] = main_result_lbl
    _upload_refs['desc_lbl_upload'] = desc_lbl_upload
    _upload_refs['progress_bars'] = progress_bars
    _upload_refs['progress_labels'] = progress_labels
    _upload_refs['upload_btn'] = upload_btn

    return widgets

# ---------------- Camera control & feed ----------------
CAMERA_UPDATE_MS = 60  # ~16 FPS -> 60 ms is lower CPU but still smooth (adjustable)

def start_camera():
    global cap, camera_running, first_camera_frame
    # Stop previous if any
    try:
        stop_camera()
    except Exception:
        pass

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        cap = cv2.VideoCapture(1)

    if not cap.isOpened():
        # no camera
        try:
            _camera_refs['camera_lbl'].configure(text="No camera connected", image="")
        except Exception:
            pass
        camera_running = False
        return

    camera_running = True
    first_camera_frame = False

    # small warm-up: run update loop
    def update_loop():
        global first_camera_frame
        if not camera_running:
            return
        if cap is None or not (hasattr(cap, "isOpened") and cap.isOpened()):
            return

        ret, frame = cap.read()
        if ret:
            if not first_camera_frame:
                first_camera_frame = True
                # hide loading if any
                hide_loading()
            # convert and resize preview to fit camera_box
            try:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pil = Image.fromarray(frame_rgb)
                box = _camera_refs.get('camera_box')
                if box:
                    w = box.winfo_width() or 320
                    h = box.winfo_height() or 240
                    pil = pil.resize((w, h), Image.BILINEAR)
                else:
                    pil = pil.resize((320, 240), Image.BILINEAR)
                imgtk = ImageTk.PhotoImage(pil)
                lbl = _camera_refs.get('camera_lbl')
                if lbl:
                    lbl.imgtk = imgtk
                    lbl.configure(image=imgtk, text="")
            except Exception as e:
                # safe fallback: ignore preview errors
                pass

        # schedule next frame
        root.after(CAMERA_UPDATE_MS, update_loop)

    # small loader show (non-blocking) then start loop
    show_loading("Starting camera...")
    def warm_task():
        time.sleep(0.4)
        root.after(0, update_loop)
    threading.Thread(target=warm_task, daemon=True).start()

def stop_camera():
    global cap, camera_running
    camera_running = False
    try:
        if cap is not None:
            try:
                cap.release()
            except Exception:
                pass
            cap = None
    except Exception:
        pass
    # clear preview
    try:
        if 'camera_lbl' in _camera_refs:
            _camera_refs['camera_lbl'].configure(image="", text="Camera stopped")
    except Exception:
        pass

# ---------------- Classification (camera & upload) ----------------
def classify_camera_frame():
    if cap is None or not (hasattr(cap, "isOpened") and cap.isOpened()):
        messagebox.showwarning("Warning", "Camera not started!")
        return

    # 🔹 SHOW processing loader
    show_loading("Processing image...")

    def do_classification():
        ret, frame = cap.read()
        if not ret:
            root.after(0, lambda: messagebox.showerror("Error", "Failed to capture image."))
            root.after(0, hide_loading)
            return

        btn = _camera_refs.get('classify_btn')
        if btn:
            root.after(0, lambda: btn.configure(state="disabled"))

        def task():
            try:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                processed = preprocess_image(frame_rgb)
                probs = model.predict_proba(processed)[0]
                top_indices = np.argsort(probs)[::-1][:4]

                root.after(0, lambda: update_camera_results(probs, top_indices))
            except Exception as e:
                root.after(0, lambda: messagebox.showerror("Error", str(e)))
            finally:
                # 🔹 HIDE loader + re-enable button
                root.after(0, lambda: hide_loading(min_visible_time=0.8))
                if btn:
                    root.after(0, lambda: btn.configure(state="normal"))

        threading.Thread(target=task, daemon=True).start()

    # keep lazy CNN loading behavior
    load_cnn(callback=do_classification)


def update_camera_results(probs, top_indices):
    top_label = CLASSES[top_indices[0]]
    top_conf = probs[top_indices[0]] * 100
    info = CLASS_INFO.get(top_label.upper(), {"name": top_label, "desc": "No description available."})

    main_lbl = _camera_refs.get('main_result_lbl_camera')
    desc_lbl = _camera_refs.get('desc_lbl_camera')
    pbars = _camera_refs.get('progress_bars_camera')
    plabels = _camera_refs.get('progress_labels_camera')
    if main_lbl:
        main_lbl.configure(text=f"{info['name']} ({top_conf:.2f}%)")
    if desc_lbl:
        desc_lbl.configure(text=f"{info['desc']}")

    # update bars & labels
    for i, idx in enumerate(top_indices):
        cname = CLASSES[idx]
        percent = probs[idx] * 100
        if i < len(plabels):
            plabels[i].configure(text=f"{cname}: ({percent:.2f}%)")
        if i < len(pbars):
            pbars[i].set(probs[idx])

    # change classify button text
    btn = _camera_refs.get('classify_btn')
    if btn:
        btn.configure(text="Classify Again")

# ---------------- Upload image flow (keeps original behavior) ----------------
def open_and_classify_image(upload_lbl, main_result_lbl, desc_lbl_upload, progress_bars, progress_labels, upload_btn):
    file_path = filedialog.askopenfilename(title="Select an Image", filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp")])
    if not file_path:
        return

    img = Image.open(file_path).convert("RGB")
    # get preview box size
    box = _upload_refs.get('upload_box')
    if box:
        bw = box.winfo_width() or 320
        bh = box.winfo_height() or 240
        img_resized = img.resize((bw, bh), Image.BILINEAR)
    else:
        img_resized = img.resize((320, 240), Image.BILINEAR)

    imgtk = ImageTk.PhotoImage(img_resized)
    upload_lbl.imgtk = imgtk
    upload_lbl.configure(image=imgtk, text="")
    uploaded_image = np.array(img_resized)

    upload_btn.configure(state="disabled")
    
    # 🔹 SHOW processing loader
    show_loading("Processing image...")

    def task():
        try:
            processed = preprocess_image(uploaded_image)
            probs = model.predict_proba(processed)[0]
            top_indices = np.argsort(probs)[::-1][:4]

            root.after(0, lambda: update_upload_results(
                probs, top_indices,
                main_result_lbl, desc_lbl_upload,
                progress_bars, progress_labels,
                upload_btn))
        except Exception as e:
            root.after(0, lambda: messagebox.showerror("Error", str(e)))
        finally:
            # 🔹 HIDE loader
            root.after(0, lambda: hide_loading(min_visible_time=0.8))

    # lazy-load CNN if needed, then run classification
    def after_load():
        threading.Thread(target=task, daemon=True).start()
    load_cnn(callback=after_load)

def update_upload_results(probs, top_indices, main_result_lbl, desc_lbl_upload, progress_bars, progress_labels, upload_btn):
    top_label = CLASSES[top_indices[0]]
    top_conf = probs[top_indices[0]] * 100
    info = CLASS_INFO.get(top_label.upper(), {"name": top_label, "desc": "No description available."})

    if main_result_lbl:
        main_result_lbl.configure(text=f"{info['name']} ({top_conf:.2f}%)")
    if desc_lbl_upload:
        desc_lbl_upload.configure(text=f"{info['desc']}")

    for i, idx in enumerate(top_indices):
        cname = CLASSES[idx]
        percent = probs[idx] * 100
        if i < len(progress_labels):
            progress_labels[i].configure(text=f"{cname} : ({percent:.2f}%)")
        if i < len(progress_bars):
            progress_bars[i].set(probs[idx])

    if upload_btn:
        upload_btn.configure(text="Upload Another Image", state="normal")

# ---------------- Lazy CNN loader (keep as before) ----------------
def load_cnn(callback=None):
    global cnn_model
    if cnn_model is not None:
        if callback:
            callback()
        return

    show_loading("Loading CNN Model...")

    def task():
        # delayed import to reduce startup time
        from tensorflow.keras.applications import EfficientNetB0
        from tensorflow.keras.models import Model
        base = EfficientNetB0(weights="imagenet", include_top=False, pooling="avg")
        globals()['cnn_model'] = Model(inputs=base.input, outputs=base.output)
        root.after(0, hide_loading)
        if callback:
            root.after(0, callback)

    threading.Thread(target=task, daemon=True).start()

# ---------------- Back from upload wrapper ----------------
def back_from_upload():
    loader_shown = [False]
    def delayed_loader():
        loader_shown[0] = True
        show_loading("Returning...")
    loader_id = root.after(200, delayed_loader)

    def task():
        # reset upload page UI if necessary
        # reset_upload_page()  # we rebuild page on show anyway
        root.after_cancel(loader_id)
        if loader_shown[0]:
            root.after(0, hide_loading)
        root.after(0, lambda: show_page(build_select_page))

    threading.Thread(target=task, daemon=True).start()

# ---------------- Run / On close ----------------
def on_close():
    try:
        stop_camera()
    except Exception:
        pass
    # ensure resources cleaned
    try:
        root.destroy()
    except Exception:
        pass

root.protocol("WM_DELETE_WINDOW", on_close)

# start on first page
show_page(build_start_page)
root.mainloop()
