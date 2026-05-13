import gradio as gr
import joblib
import numpy as np
import os

# ── Load artifacts ─────────────────────────────
MODEL_PATH = "mf_model.joblib"
DATA_PATH  = "recommender_data.joblib"

mf = data = products_df = rating_matrix = None
item_sim = user_sim = reconstructed = user_mean = users_df = None
artifacts_loaded = False

def load_artifacts():
    global mf, data, products_df, rating_matrix
    global item_sim, user_sim, reconstructed, user_mean, users_df
    global artifacts_loaded

    if not os.path.exists(MODEL_PATH) or not os.path.exists(DATA_PATH):
        return False

    mf   = joblib.load(MODEL_PATH)
    data = joblib.load(DATA_PATH)

    products_df   = data['products_df']
    rating_matrix = data['rating_matrix']
    item_sim      = data['item_sim']
    user_sim      = data['user_sim']
    reconstructed = data['reconstructed']
    user_mean     = data['user_mean']
    users_df      = data['users_df']

    artifacts_loaded = True
    return True

load_artifacts()


# ── Recommendation logic (same as your Flask version) ──
def recommend_from_cart(cart_text, top_n=8):
    if not artifacts_loaded:
        return "Model not loaded"

    try:
        cart_product_ids = [int(x.strip()) for x in cart_text.split(",") if x.strip()]
    except:
        return "Invalid input. Use comma-separated product IDs like: 1,2,3"

    cart_set = set(cart_product_ids)
    N = len(products_df)
    score_accum = np.zeros(N)

    for cid in cart_product_ids:
        score_accum += item_sim[cid]

    for cid in cart_set:
        if cid < len(score_accum):
            score_accum[cid] = 0.0

    score_accum /= max(len(cart_product_ids), 1)
    top_ids = np.argsort(score_accum)[::-1][:top_n]

    results = []
    for pid in top_ids:
        if pid >= len(products_df) or score_accum[pid] <= 0:
            continue

        row = products_df.iloc[pid]
        results.append(
            f"{row['product_name']} | {row['category']} | ${round(float(row['price']),2)}"
        )

    return "\n".join(results) if results else "No recommendations found"


# ── GRADIO UI ─────────────────────────────
demo = gr.Interface(
    fn=recommend_from_cart,
    inputs=[
        gr.Textbox(
            label="Enter Cart Product IDs (comma separated)",
            placeholder="e.g. 1,2,3"
        ),
        gr.Slider(1, 20, value=8, label="Top N Recommendations")
    ],
    outputs=gr.Textbox(label="Recommended Products"),
    title="🛍️ Product Recommendation System",
    description="Enter product IDs from your cart and get smart ML-based recommendations."
)

demo.launch()