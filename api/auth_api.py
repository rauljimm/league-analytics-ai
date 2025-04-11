# api/auth_api.py
from flask import Flask, request, jsonify
import sqlite3
import bcrypt
import os

app = Flask(__name__)
DB_PATH = os.path.join("database", "users.db")

# Crear la base de datos y la tabla si no existen
def init_db():
    if not os.path.exists("database"):
        os.makedirs("database")
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        summoner_name TEXT UNIQUE NOT NULL,
        email TEXT UNIQUE NOT NULL,
        password_hash TEXT NOT NULL
    )''')
    conn.commit()
    conn.close()

# Endpoint para registro
@app.route('/register', methods=['POST'])
def register():
    data = request.get_json()
    summoner_name = data.get('summoner_name')
    email = data.get('email')
    password = data.get('password')

    if not summoner_name or not email or not password:
        return jsonify({"error": "Todos los campos son obligatorios"}), 400
    if "@" not in email:
        return jsonify({"error": "Correo inválido"}), 400

    # Hash de la contraseña
    password_hash = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt())

    try:
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        c.execute("INSERT INTO users (summoner_name, email, password_hash) VALUES (?, ?, ?)",
                  (summoner_name, email, password_hash))
        conn.commit()
        conn.close()
        return jsonify({"message": f"¡Usuario {summoner_name} registrado con éxito!"}), 201
    except sqlite3.IntegrityError:
        return jsonify({"error": "El nombre de invocador o correo ya está en uso"}), 409
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Endpoint para login
@app.route('/login', methods=['POST'])
def login():
    data = request.get_json()
    summoner_name = data.get('summoner_name')
    password = data.get('password')

    if not summoner_name or not password:
        return jsonify({"error": "Todos los campos son obligatorios"}), 400

    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT password_hash FROM users WHERE summoner_name = ?", (summoner_name,))
    result = c.fetchone()
    conn.close()

    if result:
        password_hash = result[0]
        if bcrypt.checkpw(password.encode('utf-8'), password_hash):
            return jsonify({"message": f"¡Bienvenido de vuelta, {summoner_name}!"}), 200
        else:
            return jsonify({"error": "Contraseña incorrecta"}), 401
    else:
        return jsonify({"error": "Usuario no encontrado"}), 404

if __name__ == "__main__":
    init_db()
    app.run(debug=True, port=5000)