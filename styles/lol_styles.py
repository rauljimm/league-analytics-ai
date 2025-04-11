# styles/lol_styles.py
import customtkinter as ctk
from tkinter import font
from PIL import Image
import os

class LoLStyles:
    COLORS = {
        "background": "#0A1E3A",
        "frame": "#1B2A44",
        "border": "#FFD700",
        "text": "#FFFFFF",
        "accent": "#00FFFF",
        "button": "#FFD700",
        "button_hover": "#FFC107",
        "button_text": "#1B2A44",
        "entry_bg": "#2E3B55"
    }
    
    FONT_PATHS = {
        "Regular": os.path.join("fonts", "BeaufortForLOL-Regular.ttf"),
        "Bold": os.path.join("fonts", "BeaufortForLOL-Bold.ttf"),
        "Heavy": os.path.join("fonts", "BeaufortForLOL-Heavy.ttf"),
        "Light": os.path.join("fonts", "BeaufortForLOL-Light.ttf"),
        "LightItalic": os.path.join("fonts", "BeaufortForLOL-LightItalic.ttf")
    }
    
    @staticmethod
    def load_fonts():
        try:
            all_fonts_exist = all(os.path.exists(path) for path in LoLStyles.FONT_PATHS.values())
            if all_fonts_exist:
                font.nametofont("TkDefaultFont").configure(family="Beaufort for LOL", size=12)
                return {
                    "title": ctk.CTkFont(family="Beaufort for LOL", size=30, weight="bold"),
                    "label": ctk.CTkFont(family="Beaufort for LOL", size=16),
                    "entry": ctk.CTkFont(family="Beaufort for LOL", size=14, weight="normal"),
                    "button": ctk.CTkFont(family="Beaufort for LOL", size=18, weight="bold"),
                    "status": ctk.CTkFont(family="Beaufort for LOL", size=14, slant="italic")
                }
            else:
                print("Algunas fuentes Beaufort for LOL no encontradas, usando Arial.")
        except Exception as e:
            print(f"Error cargando fuentes: {e}")
        
        return {
            "title": ctk.CTkFont(family="Arial", size=30, weight="bold"),
            "label": ctk.CTkFont(family="Arial", size=16),
            "entry": ctk.CTkFont(family="Arial", size=14),
            "button": ctk.CTkFont(family="Arial", size=18, weight="bold"),
            "status": ctk.CTkFont(family="Arial", size=14, slant="italic")
        }
    
    @staticmethod
    def setup_background(window, image_path="assets/lol_background.jpg"):
        window.bg_label = None
        window.bg_image = None
        try:
            if os.path.exists(image_path):
                window.bg_label = ctk.CTkLabel(window, text="")
                window.bg_label.place(relwidth=1, relheight=1)
                LoLStyles.update_background(window, image_path)
            else:
                window.configure(fg_color=LoLStyles.COLORS["background"])
        except Exception as e:
            print(f"Error cargando imagen de fondo: {e}")
            window.configure(fg_color=LoLStyles.COLORS["background"])
    
    @staticmethod
    def update_background(window, image_path="assets/lol_background.jpg"):
        if not hasattr(window, "bg_label") or not window.bg_label or not os.path.exists(image_path):
            if hasattr(window, "configure"):
                window.configure(fg_color=LoLStyles.COLORS["background"])
            return
        try:
            width = window.winfo_width()
            height = window.winfo_height()
            if width < 100 or height < 100:
                return
            bg_image = Image.open(image_path)
            img_ratio = bg_image.width / bg_image.height
            win_ratio = width / height
            if win_ratio > img_ratio:
                new_width = int(height * img_ratio)
                new_height = height
            else:
                new_width = width
                new_height = int(width / img_ratio)
            bg_image = bg_image.resize((new_width, new_height), Image.LANCZOS)
            window.bg_image = ctk.CTkImage(light_image=bg_image, dark_image=bg_image, size=(new_width, new_height))
            window.bg_label.configure(image=window.bg_image)
        except Exception as e:
            print(f"Error actualizando fondo: {e}")
            window.configure(fg_color=LoLStyles.COLORS["background"])





            