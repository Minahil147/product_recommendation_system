# RecoEngine — Local Flask Web App

## Setup (once)

```bash
pip install -r requirements.txt
```

## Step 1 — Save model from Colab

Add this at the end of your notebook and run it:

```python
import joblib

joblib.dump(mf, 'mf_model.joblib')

joblib.dump({
    'user_sim':      user_sim,
    'item_sim':      item_sim,
    'reconstructed': reconstructed,
    'rating_matrix': rating_matrix,
    'user_mean':     user_mean,
    'products_df':   products_df,
    'users_df':      users_df,
}, 'recommender_data.joblib')

print("Saved!")
```

Then download both files from Colab:
```python
from google.colab import files
files.download('mf_model.joblib')
files.download('recommender_data.joblib')
```

## Step 2 — Place files

Put both `.joblib` files in the same folder as `app.py`:

```
recommendation_app/
├── app.py
├── requirements.txt
├── mf_model.joblib          ← download from Colab
├── recommender_data.joblib  ← download from Colab
└── templates/
    └── index.html
```

## Step 3 — Run

```bash
python app.py
```

Open: http://localhost:5000

## API Endpoints

| Endpoint | Description |
|----------|-------------|
| `GET /` | Web UI |
| `GET /api/recommend/<user_id>?method=hybrid&top_n=10` | Get recommendations |
| `GET /api/similar/<product_id>` | Similar products |
| `GET /api/products` | All products |
| `GET /api/stats` | Dataset stats |

### method options
- `hybrid` (default) — weighted ensemble of all models
- `mf` — SGD Matrix Factorization only
- `svd` — Truncated SVD only
- `user-cf` — User Collaborative Filtering
- `item-cf` — Item Collaborative Filtering
