# Car Damage Detection with YOLO11s

A computer-vision prototype that detects visible vehicle damage in uploaded images. The trained YOLO11s model predicts 22 damage categories; a Flask application displays annotated predictions and a simple illustrative repair-cost estimate.

> **Prototype disclaimer:** this is an educational portfolio project, not a certified vehicle-inspection system. Predictions can be wrong or incomplete. The repair-cost values are hard-coded illustrative estimates and are not quotes, market prices, or professional assessments.

## Results

The original Colab notebook records these metrics on the held-out test split:

| Metric | Test result |
|---|---:|
| Precision | 90.4% |
| Recall | 85.3% |
| mAP@50 | 84.7% |
| mAP@50–95 | 75.1% |

**Evaluation set:** 1,304 images and 1,916 labeled object instances.

### Dataset split

| Split | Images |
|---|---:|
| Train | 9,782 |
| Validation | 1,956 |
| Test | 1,304 |
| **Total** | **13,042** |

The dataset contains 22 classes and was downloaded from the [Roboflow project](https://universe.roboflow.com/fuad-nabiyev/car-damage-zxk33-ucsnt/dataset/1). The dataset metadata in the notebook lists a **CC BY 4.0** license; review the dataset page and comply with its current terms before redistribution. The dataset itself is not included in this repository.

Some categories have very few test examples. Per-class metrics for rare categories should therefore be interpreted cautiously; strong aggregate metrics do not imply uniform performance across all damage types.

## Model training

The model was initialized from pretrained `yolo11s.pt` weights and trained in Google Colab on an NVIDIA Tesla T4 GPU. The recorded configuration was:

- Ultralytics: `8.4.118`
- Epochs: `50` (with early-stopping patience `10`)
- Image size: `640`
- Batch size: `16`
- Device: CUDA
- Seed: `42`
- Workers: `2`
- Pretrained weights: `yolo11s.pt`

The cleaned notebook is in [`notebooks/car_damage_training.ipynb`](notebooks/car_damage_training.ipynb). It uses Google Colab Secrets for the Roboflow API key.

## Project structure

```text
car-damage-detection/
├── app.py
├── requirements.txt
├── README.md
├── .gitignore
├── models/
│   └── README.md       # put best.pt here locally
├── notebooks/
│   └── car_damage_training.ipynb
├── templates/
│   └── index.html
└── static/
    ├── demo/
    ├── uploads/
    └── results/
```

## Run locally

Python 3.10–3.12 is recommended for the dependency stack used by the notebook. Create and activate a virtual environment, then install dependencies:

```bash
python -m venv .venv
# Windows PowerShell
.venv\Scripts\Activate.ps1
# macOS/Linux
# source .venv/bin/activate

pip install -r requirements.txt
```

Place the trained weights at `models/best.pt`. The weight file is not tracked in Git; see [`models/README.md`](models/README.md). Then start the local app:

```bash
python app.py
```

Open http://127.0.0.1:5000. The application accepts JPG, JPEG, PNG, and WEBP images up to 10 MB. It uses CUDA when available and falls back to CPU if GPU inference fails.

Optional environment settings:

| Variable | Default | Purpose |
|---|---|---|
| `YOLO_MODEL_PATH` | `models/best.pt` | Path to model weights |
| `YOLO_DEVICE` | auto-detected | Override inference device, e.g. `cpu` or `cuda` |
| `YOLO_CONFIDENCE` | `0.15` | Detection confidence threshold |

## Security and limitations

- Never commit API keys. The training notebook reads `ROBOFLOW_API_KEY` from Colab Secrets.
- Do not commit `.env`, virtual environments, datasets, user uploads, or generated prediction images.
- The Flask app is configured for local development. Before public deployment, add authentication, request rate limiting, secure temporary-file handling, production WSGI hosting, and a suitable privacy policy.
- The current repair-cost logic adds a fixed amount for each detected object. It does not account for vehicle make/model, damage severity, labor, parts, region, or duplicate/overlapping detections. Treat it as a UI demonstration only.

## Acknowledgements

- [Ultralytics YOLO](https://github.com/ultralytics/ultralytics)
- [Roboflow dataset](https://universe.roboflow.com/fuad-nabiyev/car-damage-zxk33-ucsnt/dataset/1)
