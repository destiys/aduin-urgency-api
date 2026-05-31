from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import tensorflow as tf
import json
import numpy as np
import os

app = FastAPI(
    title="ADUIN Urgency Classification API",
    description="API untuk mengklasifikasikan tingkat urgensi keluhan masyarakat menggunakan BiLSTM",
    version="1.0.0"
)

# Jalur pencarian fleksibel untuk environment Hugging Face
paths_to_try_model = ["model/model_urgensi.keras", "model_urgensi.keras", "/code/model/model_urgensi.keras"]
paths_to_try_vocab = ["model/vocab_urgensi.json", "vocab_urgensi.json", "/code/model/vocab_urgensi.json"]

MODEL_PATH = None
VOCAB_PATH = None

for p in paths_to_try_model:
    if os.path.exists(p):
        MODEL_PATH = p
        break

for v in paths_to_try_vocab:
    if os.path.exists(v):
        VOCAB_PATH = v
        break

if not MODEL_PATH or not VOCAB_PATH:
    raise RuntimeError("File 'model_urgensi.keras' atau 'vocab_urgensi.json' gagal ditemukan.")

# 1. MEMUAT KOSAKATA & MEMBANGUN LAYER TEXT VECTORIZATION
with open(VOCAB_PATH, "r") as f:
    loaded_vocab = json.load(f)

vectorizer = tf.keras.layers.TextVectorization(
    max_tokens=5000,
    output_mode='int',
    output_sequence_length=120
)
vectorizer.set_vocabulary(loaded_vocab)

# Alias global jika internal model mencarinya
tokenizer = vectorizer

# 2. MEMBUAT ULANG ARSITEKTUR BILSTM (Sama persis dengan Google Colab kamu)
def build_clean_model():
    model = tf.keras.Sequential([
        tf.keras.layers.InputLayer(batch_shape=(None, 120)),
        tf.keras.layers.Embedding(input_dim=5000, output_dim=64, name="embedding_3"),
        tf.keras.layers.SpatialDropout1D(0.4),
        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32, return_sequences=True)),
        tf.keras.layers.GlobalMaxPooling1D(),
        tf.keras.layers.Dense(64, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001)),
        tf.keras.layers.Dropout(0.4),
        tf.keras.layers.Dense(3, activation='softmax')
    ])
    return model

# 3. LOAD BOBOT MODEL (Mengabaikan error konfigurasi JSON versi Keras)
try:
    model = build_clean_model()
    # load_weights hanya mengambil bobot angka tanpa peduli parameter quantization_config
    model.load_weights(MODEL_PATH, skip_mismatch=True)
    print("🚀 Sukses Besar! Model BiLSTM dibangun ulang dan bobot berhasil dimasukkan.")
except Exception as e:
    raise RuntimeError(f"Gagal memuat bobot model. Detail: {str(e)}")

class KeluhanRequest(BaseModel):
    teks_keluhan: str

@app.post("/predict")
def prediksi_urgensi(request: KeluhanRequest):
    if not request.teks_keluhan.strip():
        raise HTTPException(status_code=400, detail="Teks keluhan tidak boleh kosong")
        
    try:
        # Mengubah teks keluhan menjadi tensor angka indeks
        vektor_input = vectorizer([request.teks_keluhan])
        
        # Eksekusi prediksi
        raw_pred = model(vektor_input, training=False)
        pred = np.array(raw_pred[0]).flatten().tolist()
        skor = float(max(pred)) 
        
        # Penentuan label kategori berdasarkan skor probabilitas tertinggi
        if skor >= 0.86:
            hasil = "Tinggi"
            status = "🔴 DARURAT (Respon Segera)"
        elif skor >= 0.51:
            hasil = "Sedang"
            status = "🟡 WASPADA (Pantau Lokasi)"
        else:
            hasil = "Rendah"
            status = "🟢 AMAN (Tindak Lanjut Rutin)"
            
        return {
        "success": True,
        "input_text": request.teks_keluhan,
        "predictions": [
            {
                "label": hasil,            
                "status_tindakan": status, 
                "persentase": f"{skor * 100:.2f}%", 
                "probability": float(skor)
            }
        ],
        "all_probabilities": {
            "kategori_rendah": float(pred[0]),
            "kategori_sedang": float(pred[1]),
            "kategori_tinggi": float(pred[2])
        }
    }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Terjadi kesalahan saat inferensi server: {str(e)}")

@app.get("/")
def index():
    return {"status": "API ADUIN Aktif dan Berjalan"}