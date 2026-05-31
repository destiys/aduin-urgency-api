# ADUIN Urgency Classification API

API untuk klasifikasi tingkat urgensi teks keluhan/laporan masyarakat menggunakan Deep Learning (BiLSTM) + FastAPI.

## Base URL
https://destiys-urgensi-keluhan-api.hf.space


## Available Endpoints

| Method | Endpoint | Description |
| :--- | :--- | :--- |
| GET | `/` | Check API status |
| POST | `/predict` | Predict urgency level |


## Contoh Penggunaan (Predict)

### Request Body
```json
{
  "teks_keluhan": "Ada genangan air setinggi 50cm di jalan raya guntur"
}
```

### curl

```bash
curl -X 'POST' \
  'https://destiys-urgensi-keluhan-api.hf.space/predict' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
  "teks_keluhan": "Ada genangan air setinggi 50cm di jalan raya guntur"
}'
```

### Respon

{
  "status_code": 200,
  "laporan": "Ada genangan air setinggi 50cm di jalan raya guntur",
  "kategori": "Sedang",
  "persentase": "76.96%",
  "status_tindakan": "🟡 WASPADA (Pantau Lokasi)",
  "raw_probabilities": {
    "rendah": 0.7696378827095032,
    "sedang": 0.14493948221206665,
    "tinggi": 0.08542269468307495
  }
}

## API Documentation

https://destiys-urgensi-keluhan-api.hf.space/docs


## Local Development

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Run FastAPI

```bash
uvicorn app.main:app --reload