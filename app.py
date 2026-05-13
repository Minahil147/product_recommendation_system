from flask import Flask, render_template, jsonify, request
import joblib
import numpy as np
import pandas as pd
import os

app = Flask(__name__)

# ── MatrixFactorization class (must match Colab definition) ──
class MatrixFactorization:
    def __init__(self, n_users, n_items, n_factors=15, lr=0.005, reg=0.02, n_epochs=40):
        self.K, self.lr, self.reg, self.n_epochs = n_factors, lr, reg, n_epochs
        scale    = 1 / np.sqrt(n_factors)
        self.P   = np.random.normal(0, scale, (n_users,  n_factors))
        self.Q   = np.random.normal(0, scale, (n_items,  n_factors))
        self.bu  = np.zeros(n_users)
        self.bi  = np.zeros(n_items)
        self.mu  = 0.0

    def fit(self, train_data, val_data=None):
        self.mu = np.mean([r for _, _, r in train_data])
        self.train_losses, self.val_losses = [], []
        for epoch in range(self.n_epochs):
            np.random.shuffle(train_data)
            sq_loss = 0.0
            for u, i, r in train_data:
                pred     = self.mu + self.bu[u] + self.bi[i] + self.P[u] @ self.Q[i]
                err      = r - pred
                sq_loss += err ** 2
                self.bu[u] += self.lr * (err - self.reg * self.bu[u])
                self.bi[i] += self.lr * (err - self.reg * self.bi[i])
                pu_old      = self.P[u].copy()
                self.P[u]  += self.lr * (err * self.Q[i] - self.reg * self.P[u])
                self.Q[i]  += self.lr * (err * pu_old   - self.reg * self.Q[i])
            train_rmse = np.sqrt(sq_loss / len(train_data))
            self.train_losses.append(train_rmse)
            if val_data:
                val_errs = [(r - self.predict(u, i))**2 for u, i, r in val_data]
                self.val_losses.append(np.sqrt(np.mean(val_errs)))

    def predict(self, u, i):
        return float(np.clip(self.mu + self.bu[u] + self.bi[i] + self.P[u] @ self.Q[i], 1, 5))

    def recommend(self, user_id, rated_pids, top_n=10):
        scores = [(i, self.predict(user_id, i))
                  for i in range(self.Q.shape[0]) if i not in rated_pids]
        return sorted(scores, key=lambda x: x[1], reverse=True)[:top_n]


# ── Load artifacts ────────────────────────────────────────────
MODEL_PATH = 'mf_model.joblib'
DATA_PATH  = 'recommender_data.joblib'

mf = data = products_df = rating_matrix = None
item_sim = user_sim = reconstructed = user_mean = users_df = None

def load_artifacts():
    global mf, data, products_df, rating_matrix, item_sim
    global user_sim, reconstructed, user_mean, users_df

    if not os.path.exists(MODEL_PATH) or not os.path.exists(DATA_PATH):
        print("Model files not found.")
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

    print(f"Loaded | {len(users_df)} users | {len(products_df)} products")
    return True

artifacts_loaded = load_artifacts()


# ── Cart-based recommendation logic ──────────────────────────
def recommend_from_cart(cart_product_ids, top_n=8):
    cart_set    = set(cart_product_ids)
    N           = len(products_df)
    score_accum = np.zeros(N)

    for cid in cart_product_ids:
        score_accum += item_sim[cid]

    for cid in cart_set:
        score_accum[cid] = 0.0

    score_accum /= max(len(cart_product_ids), 1)
    top_ids      = np.argsort(score_accum)[::-1][:top_n]

    results = []
    for pid in top_ids:
        if score_accum[pid] <= 0:
            continue
        row = products_df[products_df.product_id == int(pid)].iloc[0]
        results.append({
            'product_id': int(pid),
            'product':    row.product_name,
            'category':   row.category,
            'price':      round(float(row.price), 2),
            'similarity': round(float(score_accum[pid]), 4),
        })
    return results


# ── Routes ────────────────────────────────────────────────────
@app.route('/')
def index():
    if not artifacts_loaded:
        return render_template('index.html', products=[], categories=[], error=True)
    prods = products_df[['product_id','product_name','category','price']].to_dict(orient='records')
    cats  = sorted(products_df['category'].unique().tolist())
    return render_template('index.html', products=prods, categories=cats, error=False)


@app.route('/api/recommend-from-cart', methods=['POST'])
def cart_recommend():
    if not artifacts_loaded:
        return jsonify({'error': 'Model not loaded'}), 503
    body = request.get_json()
    if not body or 'cart' not in body or not body['cart']:
        return jsonify({'recommendations': [], 'message': 'Cart is empty'})
    try:
        results = recommend_from_cart(body['cart'], top_n=int(body.get('top_n', 8)))
        return jsonify({'recommendations': results, 'cart_size': len(body['cart'])})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/stats')
def stats():
    if not artifacts_loaded:
        return jsonify({'error': 'Model not loaded'}), 503
    return jsonify({
        'n_products': int(len(products_df)),
        'n_ratings':  int((~rating_matrix.isna()).sum().sum()),
        'density':    round(float((~rating_matrix.isna()).sum().sum() / rating_matrix.size * 100), 1),
        'categories': int(products_df['category'].nunique()),
    })


if __name__ == '__main__':
    app.run(debug=True, port=5000)
