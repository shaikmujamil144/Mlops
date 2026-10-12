# ML CI/CD Assignment

A small scikit-learn model with automated tests and GitHub Actions deployment to GitHub Pages.

## Project details
- **Model:** Decision Tree Classifier
- **Dataset:** Iris dataset included with scikit-learn
- **Tests:** pytest
- **Pipeline:** CI trains and tests the model; CD runs only after CI succeeds and publishes `index.html` to GitHub Pages.

## Files
- `model.py` — loads Iris, trains the model, makes a prediction, and prints test accuracy.
- `test_model.py` — checks training, accuracy bounds, and prediction output.
- `requirements.txt` — Python dependencies.
- `.github/workflows/cicd.yml` — CI/CD workflow with `needs: ci` for deployment gating.
- `index.html` — static project status page.

## GitHub setup
1. Create a **public** repository named `ML-CICD-Assignment` and add these files to its default `main` branch.
2. Open **Settings → Pages** and set **Build and deployment → Source** to **GitHub Actions**.
3. Open **Actions**, select **ML Model CI/CD**, and confirm the workflow completes successfully. A push to `main` or manual workflow dispatch will run it.
4. After a successful deployment, the Pages site URL is shown in the workflow run and under **Settings → Pages**.

## Run locally (optional)
```bash
python -m pip install -r requirements.txt
python model.py
pytest -q
```
No local Git installation is needed to create or edit the repository through the GitHub website.
