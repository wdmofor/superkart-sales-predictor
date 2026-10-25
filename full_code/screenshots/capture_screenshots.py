"""Capture screenshots of the running local SuperKart containers.

Run with the containers from section 10 already healthy (local backup run):

    python screenshots/capture_screenshots.py

Public Codespace run, using the links in the notebook's Links table:

    python screenshots/capture_screenshots.py \
        --backend-url https://<codespace>-7860.app.github.dev/ \
        --frontend-url https://<codespace>-7870.app.github.dev/ \
        --out-dir screenshots/public

Requires Playwright for Python and a local Chrome at CHROME_PATH.
Writes four PNG files and capture_record.json into the output folder.
"""
import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

from playwright.sync_api import sync_playwright

OUT = Path(__file__).resolve().parent
FULL_CODE = OUT.parent
BATCH_CSV = FULL_CODE.parent / "data" / "Batch_Data_SuperKart.csv"
MODEL_FILE = FULL_CODE / "backend_files" / "superkart_model.joblib"
CHROME_PATH = "/usr/bin/google-chrome"

PAYLOAD = {
    "Product_Weight": 12.66,
    "Product_Sugar_Content": "Low Sugar",
    "Product_Allocated_Area": 0.027,
    "Product_MRP": 117.08,
    "Store_Size": "Medium",
    "Store_Location_City_Type": "Tier 2",
    "Store_Type": "Supermarket Type2",
    "Product_Id_char": "FD",
    "Store_Age_Years": 16,
    "Product_Type_Category": "Non Perishables",
}

RENDER_JS = """
async (payload) => {
  const res = await fetch('/v1/predict', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify(payload)
  });
  const text = await res.text();
  document.body.innerHTML = '';
  document.body.style.fontFamily = 'sans-serif';
  document.body.style.margin = '16px';
  const h = document.createElement('h3');
  h.textContent = 'POST ' + location.origin + '/v1/predict (HTTP ' + res.status + ')';
  const req = document.createElement('pre');
  req.textContent = JSON.stringify(payload, null, 2);
  const out = document.createElement('pre');
  out.textContent = JSON.stringify(JSON.parse(text), null, 2);
  document.body.append(h, req, out);
  return {status: res.status, body: text};
}
"""


PORT_WARNING_TITLE = "Codespaces Access Port"


def goto_app(page, url: str) -> None:
    """Open an app URL, clicking through GitHub's port warning page if it appears."""
    page.goto(url)
    if page.title() == PORT_WARNING_TITLE:
        with page.expect_navigation():
            page.get_by_text("Continue", exact=True).click()
    page.wait_for_load_state("load")


def main() -> None:
    parser = argparse.ArgumentParser(description="Capture SuperKart app screenshots.")
    parser.add_argument("--backend-url", default="http://127.0.0.1:7860/")
    parser.add_argument("--frontend-url", default="http://127.0.0.1:7870/")
    parser.add_argument("--out-dir", default=str(OUT))
    args = parser.parse_args()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    with sync_playwright() as p:
        browser = p.chromium.launch(executable_path=CHROME_PATH, headless=True)
        page = browser.new_page(viewport={"width": 1000, "height": 260})

        goto_app(page, args.backend_url)
        page.wait_for_load_state("networkidle")
        page.screenshot(path=str(out / "backend_root.png"))

        page.set_viewport_size({"width": 900, "height": 620})
        result = page.evaluate(RENDER_JS, PAYLOAD)
        assert result["status"] == 200, result
        template_prediction = json.loads(result["body"])["Predicted_Product_Store_Sales_Total"]
        assert f"{template_prediction:.2f}" == "2904.05", result
        page.screenshot(path=str(out / "backend_predict_response.png"))
        print("backend response:", result["body"].strip())

        page.set_viewport_size({"width": 1280, "height": 1150})
        goto_app(page, args.frontend_url)
        page.get_by_role("button", name="Predict sales").wait_for(timeout=60000)
        page.get_by_role("button", name="Predict sales").click()
        page.get_by_text("Predicted Product_Store_Sales_Total").wait_for(timeout=60000)
        assert page.get_by_text("2,904.05").count() >= 1
        page.screenshot(path=str(out / "frontend_single_prediction.png"))
        print("frontend single prediction: 2,904.05")

        page.get_by_role("tab", name="Batch CSV").click()
        page.locator('input[type="file"]').first.set_input_files(str(BATCH_CSV))
        page.get_by_role("button", name="Predict batch").wait_for(timeout=60000)
        page.get_by_role("button", name="Predict batch").click()
        page.get_by_text("Rows received: 10, predicted: 8").wait_for(timeout=60000)
        page.locator('[data-testid="stDataFrame"]').nth(2).wait_for(timeout=60000)
        page.wait_for_timeout(3000)
        page.set_viewport_size({"width": 1280, "height": 1300})
        page.wait_for_timeout(1000)
        page.screenshot(path=str(out / "frontend_batch_tab.png"), full_page=True)
        print("frontend batch: Rows received: 10, predicted: 8")
        browser.close()

    record = {
        "captured_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "backend_url": args.backend_url,
        "frontend_url": args.frontend_url,
        "model_sha256": hashlib.sha256(MODEL_FILE.read_bytes()).hexdigest(),
        "template_prediction": template_prediction,
        "frontend_single_prediction_text": "2,904.05",
        "batch_rows_received": 10,
        "batch_rows_predicted": 8,
    }
    (out / "capture_record.json").write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    print("wrote capture_record.json")


if __name__ == "__main__":
    main()
