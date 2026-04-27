from flask import Flask, request, jsonify, render_template, session, redirect, url_for
import pandas as pd
import pymysql
import bcrypt
from chatbot import chat_gemini  # import chatbot Gemini

# =========================
# KONFIGURASI DATABASE MySQL
# =========================
DB_CONFIG = {
    "host":        "localhost",
    "user":        "root",
    "password":    "",
    "database":    "wisata_bali",
    "charset":     "utf8mb4",
    "cursorclass": pymysql.cursors.DictCursor
}


def get_db():
    return pymysql.connect(**DB_CONFIG)


def resolve_user_table(conn):
    return "users"


# =========================
# LOAD DATASET
# =========================
df = pd.read_csv("dataset_wisata_bali.csv")

app = Flask(__name__)
app.secret_key = "secret123"


# =========================
# REGISTER
# =========================
@app.route('/register', methods=['POST'])
def register():
    try:
        data = request.get_json(silent=True) or {}

        nama     = data['nama']
        gmail    = data['gmail']
        password = data['password']

        hashed = bcrypt.hashpw(
            password.encode('utf-8'), bcrypt.gensalt()
        ).decode('utf-8')

        conn = get_db()
        try:
            user_table = resolve_user_table(conn)
            with conn.cursor() as c:
                c.execute(f"SELECT id FROM {user_table} WHERE nama = %s", (nama,))
                if c.fetchone():
                    return jsonify({"message": "Nama sudah digunakan"}), 400

                c.execute(f"SELECT id FROM {user_table} WHERE gmail = %s", (gmail,))
                if c.fetchone():
                    return jsonify({"message": "Gmail sudah digunakan"}), 400

                c.execute(
                    f"INSERT INTO {user_table} (nama, gmail, password, role) VALUES (%s, %s, %s, %s)",
                    (nama, gmail, hashed, 'user')
                )
            conn.commit()
        finally:
            conn.close()

        return jsonify({"message": "Register berhasil"})

    except KeyError:
        return jsonify({"message": "Data register tidak lengkap"}), 400
    except Exception as e:
        print("REGISTER ERROR:", e)
        return jsonify({"message": "Terjadi error"}), 500


# =========================
# LOGIN
# =========================
@app.route('/login', methods=['POST'])
def login():
    try:
        data = request.get_json(silent=True) or {}

        nama     = data['nama']
        password = data['password']

        conn = get_db()
        try:
            user_table = resolve_user_table(conn)
            with conn.cursor() as c:
                c.execute(
                    f"SELECT * FROM {user_table} WHERE nama = %s",
                    (nama,)
                )
                user = c.fetchone()
        finally:
            conn.close()

        if user:
            stored_password = user['password'].encode('utf-8')
            if bcrypt.checkpw(password.encode('utf-8'), stored_password):
                session['user_id'] = user['id']
                session['role']    = user.get('role', 'user')
                return jsonify({"message": "Login berhasil"})

        return jsonify({"message": "Nama atau password salah"}), 401

    except KeyError:
        return jsonify({"message": "Nama dan password wajib diisi"}), 400
    except Exception as e:
        print("LOGIN ERROR:", e)
        return jsonify({"message": "Terjadi error login"}), 500


# =========================
# SESSION CHECK
# =========================
@app.route('/me')
def me():
    if 'user_id' in session:
        return jsonify({"user_id": session['user_id'], "role": session['role']})
    return jsonify({"message": "Belum login"}), 401


# =========================
# LOGOUT
# =========================
@app.route('/logout')
def logout():
    session.clear()
    return jsonify({"message": "Logout berhasil"})


# =========================
# HALAMAN LOGIN
# =========================
@app.route("/login-page")
def login_page():
    if 'user_id' in session:
        return redirect(url_for('ui'))
    return render_template("login.html")


# =========================
# UI (PROTECTED)
# =========================
@app.route("/ui")
def ui():
    if 'user_id' not in session:
        return redirect(url_for('login_page'))
    return render_template("index.html")


# =========================
# HOME
# =========================
@app.route("/")
def home():
    return "API Rekomendasi Wisata Bali Aktif 🚀"


# =========================
# KATEGORI
# =========================
@app.route("/kategori")
def get_kategori():
    kategori_list = sorted(df['kategori'].dropna().unique().tolist())
    return jsonify(kategori_list)


# =========================
# REKOMENDASI (PROTECTED)
# =========================
@app.route("/rekomendasi")
def get_rekomendasi():
    if 'user_id' not in session:
        return jsonify({"message": "Harus login dulu"}), 401

    tempat   = request.args.get("tempat",   "").strip()
    kategori = request.args.get("kategori", "").strip()

    hasil = df.copy()

    if tempat:
        hasil = hasil[hasil['nama_tempat'].str.contains(tempat, case=False, na=False)]
    if kategori:
        hasil = hasil[hasil['kategori'].str.contains(kategori, case=False, na=False)]

    hasil = hasil.sort_values(by="rating", ascending=False)
    hasil = hasil.dropna(subset=['latitude', 'longitude'])

    if hasil.empty:
        return jsonify({"message": "Tempat tidak ditemukan"})

    data = []
    for _, row in hasil.iterrows():
        try:
            lat = float(row['latitude'])
            lng = float(row['longitude'])
        except Exception:
            continue

        data.append({
            "nama_tempat": row.get("nama_tempat", ""),
            "kategori":    row.get("kategori",    ""),
            "kabupaten":   row.get("kabupaten",   ""),
            "rating":      float(row.get("rating", 0)),
            "latitude":    lat,
            "longitude":   lng,
            "deskripsi":   row.get("deskripsi",   ""),
            "aktivitas":   row.get("aktivitas",   "")
        })

    return jsonify(data)


# =========================
# CHATBOT (PROTECTED)
# =========================
@app.route("/chatbot", methods=["POST"])
def chatbot():
    if 'user_id' not in session:
        return jsonify({"message": "Harus login dulu"}), 401

    try:
        data    = request.get_json(silent=True) or {}
        pesan   = data.get("pesan", "").strip()
        riwayat = data.get("riwayat", [])

        if not pesan:
            return jsonify({"balasan": "Pesan tidak boleh kosong."}), 400

        balasan = chat_gemini(pesan, riwayat)
        return jsonify({"balasan": balasan})

    except Exception as e:
        print("CHATBOT ERROR:", e)
        return jsonify({"balasan": "Maaf, terjadi kesalahan pada chatbot."}), 500


# =========================
# RUN
# =========================
if __name__ == "__main__":
    app.run(debug=True)