# Light Object Detection API

A lightweight Python API for object detection and face recognition with pluggable backends. It is designed to be simple to run beside NVR software such as lightNVR while keeping object detection, face matching, and face crop extraction in a separate service.

## Features

- FastAPI-based REST API for object detection
- Pluggable backend architecture for different detection engines
- ONNX/YOLO object detection by default, with optional TensorFlow Lite support
- Support for image uploads and detection with confidence thresholds
- InsightFace-based face training, recognition, and face crop detection
- SQLite-backed local face embedding database
- Extensible design for adding new detection backends

## Requirements

- Python 3.9+
- pipenv (for dependency management)

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/light-object-detect.git
   cd light-object-detect
   ```

2. Install dependencies:
   ```bash
   pipenv install
   ```

3. (Optional) Pre-download the default TFLite model:
   ```bash
   pipenv run python scripts/download_model.py
   ```

   If you skip this step, the API will try to download the default model on startup when the `tflite` backend is enabled (requires internet access). Docker builds download the default model by default.

## Usage

1. Start the API server using the provided script:
   ```bash
   pipenv run python scripts/run_server.py --reload
   ```
   
   Or manually with uvicorn:
   ```bash
   pipenv run uvicorn main:app --reload --port 9001
   ```

2. The API will be available at http://localhost:9001

3. Access the API documentation at http://localhost:9001/docs

## Docker (e.g. Unraid / lightNVR)

### Build

```bash
docker build -t light-object-detect:local .
```

By default, the image downloads a small reference TFLite model at build time so the `tflite` backend works out of the box.
To disable this, build with `--build-arg DOWNLOAD_DEFAULT_MODEL=0`.

### Run

Option A: without `.env` (uses defaults from `config.py`):

```bash
docker run --rm -p 8000:8000 --name light-object-detect light-object-detect:local
```

Option B: with `.env` (recommended, e.g. for backend/model paths):

```bash
docker run --rm -p 8000:8000 --name light-object-detect \
  -v "$(pwd)/.env:/app/.env:ro" \
  light-object-detect:local
```

PowerShell:

```powershell
docker run --rm -p 8000:8000 --name light-object-detect `
  -v "${PWD}\.env:/app/.env:ro" `
  light-object-detect:local
```

- **Healthcheck**: `GET /health`
- **Swagger UI**: `GET /docs`

### lightNVR Integration

In lightNVR, the object detection API URL is typically:

- `http://<docker-host>:8000/api/v1/detect`

For face recognition in lightNVR, use:

- `http://<docker-host>:8000/api/v1/faces/recognize`

When running both services in the same Docker Compose project, the service name can be used instead:

- `http://light-object-detect:8000/api/v1/detect`
- `http://light-object-detect:8000/api/v1/faces/recognize`

Face embeddings are stored in `data/faces.db` by default. Mount `/app/data` as a Docker volume if you want trained faces to survive container rebuilds.

Face crop quality can be tuned with environment variables:

- `FACE_DET_SIZE` - InsightFace detector input size. Default: `960`; raise toward `1280` for smaller/distant faces if CPU allows.
- `FACE_CROP_PADDING_RATIO` - Extra context around the detected face. Default: `0.6`.
- `FACE_CROP_MIN_SIZE` - Small crops are upscaled to at least this size for review/training. Default: `320`.
- `FACE_CROP_MAX_SIZE` - Large crops are downscaled to cap response/storage size. Default: `768`.
- `FACE_CROP_JPEG_QUALITY` - JPEG quality for returned crops. Default: `95`.

## API Endpoints

- `GET /` - Root endpoint with API information
- `GET /health` - Health check endpoint (useful for Docker/Unraid)
- `GET /api/v1/backends` - List available detection backends
- `POST /api/v1/detect` - Detect objects in an uploaded image
- `POST /api/v1/faces/train` - Train a known face from an uploaded image
- `POST /api/v1/faces/recognize` - Recognize the most prominent face in an uploaded image
- `POST /api/v1/faces/detect` - Detect all faces and return normalized boxes plus JPEG crops
- `GET /api/v1/faces/list` - List trained faces
- `DELETE /api/v1/faces/{face_id}` - Delete a trained face

### Example: Detect objects in an image

```bash
curl -X POST "http://localhost:9001/api/v1/detect" \
  -H "accept: application/json" \
  -F "file=@/path/to/your/image.jpg" \
  -F "backend=onnx" \
  -F "confidence_threshold=0.5"
```

### Example: Train a face

```bash
curl -X POST "http://localhost:9001/api/v1/faces/train" \
  -F "name=Alice" \
  -F "file=@/path/to/alice.jpg"
```

### Example: Recognize a face

```bash
curl -X POST "http://localhost:9001/api/v1/faces/recognize" \
  -F "file=@/path/to/snapshot.jpg"
```

Example response:

```json
{
  "success": true,
  "name": "Alice",
  "confidence": 0.71,
  "process_time_ms": 320
}
```

If no trained face matches, the API returns `success: false` and `name: "Unknown"`.

### Example: Detect face boxes and crops

```bash
curl -X POST "http://localhost:9001/api/v1/faces/detect" \
  -F "file=@/path/to/snapshot.jpg"
```

Example response:

```json
{
  "success": true,
  "face_count": 1,
  "faces": [
    {
      "index": 0,
      "bbox": {
        "x": 0.32,
        "y": 0.18,
        "width": 0.14,
        "height": 0.22
      },
      "confidence": 0.86,
      "crop_width": 320,
      "crop_height": 320,
      "crop_jpeg_base64": "..."
    }
  ],
  "process_time_ms": 95
}
```

## Adding New Backends

To add a new detection backend:

1. Create a new module in the `backends` directory
2. Implement the `DetectionBackend` interface
3. Register the backend in `backends/factory.py`

## Project Structure

```
light-object-detect/
├── api/                    # API endpoints
│   ├── v1/                 # API version 1
│   │   └── endpoints/      # API endpoints
│   │       ├── detection.py # Object detection endpoints
│   │       └── recognition.py # Face recognition endpoints
│   └── router.py           # API router
├── backends/               # Detection backends
│   ├── base.py             # Base backend interface
│   ├── factory.py          # Backend factory
│   ├── face/               # InsightFace recognition engine
│   └── tflite/             # TFLite backend
│       └── backend.py      # TFLite implementation
├── models/                 # Data models
│   └── detection.py        # Detection models
├── scripts/                # Utility scripts
│   ├── download_model.py   # Script to download models
│   ├── run_server.py       # Script to run the API server
│   └── test_api.py         # Script to test the API
├── utils/                  # Utility functions
│   ├── face_db.py          # SQLite face embedding database
│   └── image.py            # Image processing utilities
├── config.py               # Application configuration
├── main.py                 # FastAPI application
├── Pipfile                 # Dependencies
└── README.md               # This file
```

## License

Licensed under GPLv3
