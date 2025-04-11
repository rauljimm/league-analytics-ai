# views/register_view.py
import customtkinter as ctk
import requests
import threading
import queue
from styles.lol_styles import LoLStyles

class RegisterView(ctk.CTk):
    def __init__(self, switch_to_login_callback=None):
        super().__init__()
        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("blue")
        self.title("Registro - League of Legends")
        self.geometry("700x600")
        self.resizable(True, True)
        self.minsize(400, 300)
        self.switch_to_login_callback = switch_to_login_callback
        
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(0, weight=1)
        
        self.styles = LoLStyles()
        self.fonts = self.styles.load_fonts()
        LoLStyles.setup_background(self)
        
        self.main_frame = ctk.CTkFrame(self, corner_radius=15, fg_color=self.styles.COLORS["frame"],
                                       border_width=2, border_color=self.styles.COLORS["border"])
        self.main_frame.grid(row=0, column=0, padx=20, pady=20, sticky="nsew")
        self.main_frame.grid_columnconfigure(0, weight=1)
        self.main_frame.grid_rowconfigure([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], weight=1)
        
        self.title_label = ctk.CTkLabel(self.main_frame, text="CREAR CUENTA", font=self.fonts["title"],
                                        text_color=self.styles.COLORS["border"])
        self.title_label.grid(row=0, column=0, pady=(20, 10))
        
        self.username_label = ctk.CTkLabel(self.main_frame, text="Nombre de Invocador", font=self.fonts["label"],
                                           text_color=self.styles.COLORS["text"])
        self.username_label.grid(row=1, column=0, pady=(10, 5), sticky="w", padx=30)
        self.username_entry = ctk.CTkEntry(self.main_frame, height=40, corner_radius=10,
                                           placeholder_text="Ingrese su nombre de invocador", font=self.fonts["entry"],
                                           fg_color=self.styles.COLORS["entry_bg"], text_color=self.styles.COLORS["text"])
        self.username_entry.grid(row=2, column=0, pady=5, padx=30, sticky="ew")
        
        self.email_label = ctk.CTkLabel(self.main_frame, text="Correo Electrónico", font=self.fonts["label"],
                                        text_color=self.styles.COLORS["text"])
        self.email_label.grid(row=3, column=0, pady=(15, 5), sticky="w", padx=30)
        self.email_entry = ctk.CTkEntry(self.main_frame, height=40, corner_radius=10,
                                        placeholder_text="Ingrese su correo", font=self.fonts["entry"],
                                        fg_color=self.styles.COLORS["entry_bg"], text_color=self.styles.COLORS["text"])
        self.email_entry.grid(row=4, column=0, pady=5, padx=30, sticky="ew")
        
        self.password_label = ctk.CTkLabel(self.main_frame, text="Contraseña", font=self.fonts["label"],
                                           text_color=self.styles.COLORS["text"])
        self.password_label.grid(row=5, column=0, pady=(15, 5), sticky="w", padx=30)
        self.password_entry = ctk.CTkEntry(self.main_frame, height=40, corner_radius=10,
                                           placeholder_text="Ingrese su contraseña", show="*",
                                           font=self.fonts["entry"], fg_color=self.styles.COLORS["entry_bg"],
                                           text_color=self.styles.COLORS["text"])
        self.password_entry.grid(row=6, column=0, pady=5, padx=30, sticky="ew")
        
        self.register_button = ctk.CTkButton(self.main_frame, text="REGISTRARSE", command=self.register_callback,
                                             height=45, corner_radius=10, font=self.fonts["button"],
                                             fg_color=self.styles.COLORS["button"], hover_color=self.styles.COLORS["button_hover"],
                                             text_color=self.styles.COLORS["button_text"])
        self.register_button.grid(row=7, column=0, pady=20)
        
        self.status_label = ctk.CTkLabel(self.main_frame, text="", font=self.fonts["status"],
                                         text_color=self.styles.COLORS["accent"])
        self.status_label.grid(row=8, column=0, pady=(10, 10))
        
        self.login_link = ctk.CTkButton(self.main_frame, text="¿Ya tienes cuenta? Inicia sesión",
                                        command=self.switch_to_login, font=self.fonts["label"],
                                        fg_color="transparent", text_color=self.styles.COLORS["accent"],
                                        hover_color=self.styles.COLORS["frame"])
        self.login_link.grid(row=9, column=0, pady=10)
        
        self.bind("<Configure>", self.on_resize)
        
        self.response_queue = queue.Queue()
        self.running = True  # Bandera para controlar check_queue
        self.check_queue()
    
    def register_callback(self):
        username = self.username_entry.get()
        email = self.email_entry.get()
        password = self.password_entry.get()
        
        if not username or not email or not password:
            self.status_label.configure(text="Complete todos los campos.")
            return
        
        self.status_label.configure(text="Cargando...")
        self.register_button.configure(state="disabled")
        
        thread = threading.Thread(target=self._register_request, args=(username, email, password))
        thread.start()
    
    def _register_request(self, username, email, password):
        try:
            response = requests.post("http://localhost:5000/register", 
                                    json={"summoner_name": username, "email": email, "password": password}, timeout=5)
            data = response.json()
            message = data["message"] if response.status_code == 201 else data["error"]
        except requests.exceptions.RequestException as e:
            message = f"Error de conexión: {str(e)}"
        
        self.response_queue.put(message)
    
    def check_queue(self):
        if not self.running:
            return
        try:
            while not self.response_queue.empty():
                message = self.response_queue.get_nowait()
                self._update_status(message)
        except queue.Empty:
            pass
        self.after(100, self.check_queue)
    
    def _update_status(self, message):
        self.status_label.configure(text=message)
        self.register_button.configure(state="normal")
    
    def switch_to_login(self):
        if self.switch_to_login_callback:
            self.switch_to_login_callback()
    
    def on_resize(self, event):
        self.styles.update_background(self, image_path="assets/lol_background.jpg")
        width = self.winfo_width()
        scale_factor = max(0.8, min(width / 700, 1.5))
        self.fonts["title"].configure(size=int(30 * scale_factor))
        self.fonts["label"].configure(size=int(16 * scale_factor))
        self.fonts["entry"].configure(size=int(14 * scale_factor))
        self.fonts["button"].configure(size=int(18 * scale_factor))
        self.fonts["status"].configure(size=int(14 * scale_factor))
        entry_width = int(max(250, min(width * 0.6, 400)))
        self.username_entry.configure(width=entry_width)
        self.email_entry.configure(width=entry_width)
        self.password_entry.configure(width=entry_width)
        button_width = int(max(150, min(width * 0.4, 250)))
        self.register_button.configure(width=button_width)
    
    def show(self):
        self.deiconify()
    
    def hide(self):
        self.withdraw()
    
    def destroy(self):
        self.running = False 
        super().destroy()