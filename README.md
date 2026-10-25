# SuperKart Sales Predictor

Full-code submission for the SuperKart sales-prediction case study. It contains a
model notebook, a Flask API, a Streamlit app, and two Docker containers that run in a
GitHub Codespace.

## Public links

- Backend API: https://weary-tomb-v7gq949745j26647-7860.app.github.dev
- Frontend app: https://weary-tomb-v7gq949745j26647-7870.app.github.dev

These links open while the Codespace `weary-tomb-v7gq949745j26647` is running and its
ports 7860 and 7870 are set to Public.

## Repository layout

- `.devcontainer/`: Codespace configuration. It installs Docker and forwards ports 7860 and 7870.
- `data/`: `SuperKart.csv` (training data) and `Batch_Data_SuperKart.csv` (batch test file).
- `full_code/`: the notebook (`SuperKart_Full_Code.ipynb`, with its HTML export), the backend and
  frontend files, the tests, the screenshots, and `deploy_containers.sh`.

## Run in a Codespace

```bash
bash full_code/deploy_containers.sh
```

## Run the notebook

Open `full_code/SuperKart_Full_Code.ipynb`. It reads the data from `../data/`.
See `full_code/README.md` for the local Docker run and the tests.
