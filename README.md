# SuperKart sales predictor: full-code deployment

This repository holds the full-code submission for the SuperKart sales forecast.
The project is in [full_code/](full_code/). Start with
[full_code/README.md](full_code/README.md).

- `full_code/SuperKart_Full_Code.ipynb`: the notebook (EDA, preprocessing,
  model building, tuning, comparison, serialization, deployment).
- `full_code/SuperKart_Full_Code.html`: the executed notebook as HTML.
- `full_code/backend_files/`: Flask prediction API, model, metadata, Dockerfile.
- `full_code/frontend_files/`: Streamlit app, Dockerfile.
- `full_code/deploy_containers.sh`: builds and starts both containers on the
  `superkart-net` Docker network and runs a smoke test.
- `.devcontainer/`: GitHub Codespace configuration with Docker enabled.
- `data/`: the two course CSV files read by the notebook.

## Run in a GitHub Codespace

Open the repository in a Codespace, then run in its terminal:

```bash
bash full_code/deploy_containers.sh
```

The API listens on port 7860 and the app on port 7870. Both ports are private
to the Codespace and are opened from the Ports panel.
