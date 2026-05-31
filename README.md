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
```json
{
  "success": true,
  "input_text": "Ada genangan air setinggi 50cm di jalan raya guntur",
  "predictions": [
    {
      "label": "Sedang",
      "status_tindakan": "🟡 WASPADA (Pantau Lokasi)",
      "persentase": "76.96%",
      "probability": 0.7696
    }
  ],
  "all_probabilities": {
    "kategori_rendah": 0.7696,
    "kategori_sedang": 0.1449,
    "kategori_tinggi": 0.0854
  }
}
```

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