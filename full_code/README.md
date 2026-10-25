# SuperKart Sales Prediction: Full-Code Deployment

This folder holds the full-code submission. It predicts `Product_Store_Sales_Total`
with a tuned Random Forest pipeline, serves it through a Flask API, and provides a
Streamlit app that calls the API. Both run as Docker containers on a shared network.

## Layout

| Path | Contents |
| ---- | -------- |
| `SuperKart_Full_Code.ipynb` | The executed notebook: EDA, preprocessing, modelling, tuning, selection, serialization, deployment, and live checks |
| `SuperKart_Full_Code.html` | The same notebook exported to HTML |
| `backend_files/` | Flask API (`app.py`), `Dockerfile`, `requirements.txt`, and the serialized model `superkart_model.joblib` |
| `frontend_files/` | Streamlit app (`app.py`), `Dockerfile`, and `requirements.txt` |
| `tests/` | Pytest suites for the backend and the frontend |
| `deploy_containers.sh` | Builds, starts, health-checks, and smoke-tests both containers |

`.devcontainer/` sits one level up, at the repository root beside `data/` and `full_code/`. It configures a GitHub Codespace with Docker.

## Data

The notebook reads the course CSV files from `../data/`:

- `SuperKart.csv`
- `Batch_Data_SuperKart.csv`

When this folder is published as its own repository, place those two files in a
`data/` folder next to `full_code/`. The repository root must then contain
`data/`, `full_code/`, and `.devcontainer/`.

## Run the containers

Requirements: Docker. Run from the repository root or from this folder:

```bash
bash full_code/deploy_containers.sh   # from the repository root
bash deploy_containers.sh             # from full_code/
```

The script:

1. Creates the `superkart-net` Docker network if it does not exist.
2. Replaces any existing `superkart-full-backend` and `superkart-full-frontend` containers.
3. Builds both images and starts them. Ports bind to `127.0.0.1` only.
4. Waits for both health checks and sends one prediction through the API.

Then open:

- Backend API: http://127.0.0.1:7860
- Frontend app: http://127.0.0.1:7870

Stop the containers with `docker rm -f superkart-full-backend superkart-full-frontend`.

## Run in a GitHub Codespace

1. Push the repository root (with `data/`, `full_code/`, and `.devcontainer/`) to GitHub.
2. Open the repository in a Codespace. The `.devcontainer` configuration installs Docker.
3. In the Codespace terminal, run `bash full_code/deploy_containers.sh`.
4. Set ports 7860 and 7870 to Public, from the Ports panel or with `gh codespace ports visibility 7860:public 7870:public -c <codespace>`. Ports are private by default.
5. The public links are `https://<codespace>-7860.app.github.dev` (backend) and `https://<codespace>-7870.app.github.dev` (frontend). They open while the Codespace is running.

## Run the tests

From this folder:

```bash
python -m pytest tests -q -p no:cacheprovider --color=no
```

The tests need the packages in `backend_files/requirements.txt` and
`frontend_files/requirements.txt`, plus `pytest`.

## API endpoints

| Method | Path | Body | Response |
| ------ | ---- | ---- | -------- |
| GET | `/` | none | Health status and model name |
| POST | `/v1/predict` | JSON object with the 10 model features | `Predicted_Product_Store_Sales_Total` and `warnings` |
| POST | `/v1/predictbatch` | multipart form, CSV file in field `file` | `predictions` keyed by 1-based row number, plus `errors` and `warnings` |

Example:

```bash
curl -X POST http://127.0.0.1:7860/v1/predict \
  -H "Content-Type: application/json" \
  -d '{"Product_Weight": 12.66, "Product_Sugar_Content": "Low Sugar",
       "Product_Allocated_Area": 0.027, "Product_MRP": 117.08,
       "Store_Size": "Medium", "Store_Location_City_Type": "Tier 2",
       "Store_Type": "Supermarket Type2", "Product_Id_char": "FD",
       "Store_Age_Years": 16, "Product_Type_Category": "Non Perishables"}'
```

The shipped model returns 2904.05 for this record. The course template gives no expected output, so this is the model's output for the template payload, not a target value.
