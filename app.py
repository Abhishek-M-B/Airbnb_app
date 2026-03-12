from flask import Flask, render_template, request, jsonify
import pickle
import numpy as np
from datetime import datetime
import os

app = Flask(__name__)

# Load model
model = None
model_path = 'airbnb_xgboost_model.pkl'

if os.path.exists(model_path):
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    print("✅ Model loaded successfully!")
else:
    print("⚠️ Model file not found — predictions will use demo mode")

# ── Encoding maps (EXACT from your notebook LabelEncoder) ──
neighbourhood_map = {
    'Bronx'        : 0,
    'Brooklyn'     : 1,
    'Manhattan'    : 2,
    'Queens'       : 3,
    'Staten Island': 4
}

property_type_map = {
    'Apartment'      : 0,
    'Bed & Breakfast': 1,
    'Boat'           : 2,
    'Bungalow'       : 3,
    'Cabin'          : 4,
    'Camper/RV'      : 5,
    'Castle'         : 6,
    'Chalet'         : 7,
    'Condominium'    : 8,
    'Dorm'           : 9,
    'House'          : 10,
    'Hut'            : 11,
    'Lighthouse'     : 12,
    'Loft'           : 13,
    'Other'          : 14,
    'Tent'           : 15,
    'Townhouse'      : 16,
    'Treehouse'      : 17,
    'Villa'          : 18
}

# Ordinal encoding from your notebook
room_type_map = {
    'Shared room'    : 0,
    'Private room'   : 1,
    'Entire home/apt': 2
}

@app.route('/')
def index():
    return render_template('index.html',
                           neighbourhoods=list(neighbourhood_map.keys()),
                           property_types=list(property_type_map.keys()),
                           room_types=list(room_type_map.keys()))

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()

        # ── Extract inputs ──────────────────────────────
        neighbourhood   = data.get('neighbourhood', 'Manhattan')
        property_type   = data.get('property_type', 'Apartment')
        room_type       = data.get('room_type', 'Entire home/apt')
        beds            = float(data.get('beds', 1))
        review_score    = float(data.get('review_score', 90))
        host_since      = data.get('host_since', '2015-01-01')
        reviews_per_day = float(data.get('reviews_per_day', 0.1))

        # ── Feature Engineering (same as notebook) ──────
        host_since_dt    = datetime.strptime(host_since, '%Y-%m-%d')
        host_month       = host_since_dt.month
        host_days_active = (datetime.today() - host_since_dt).days

        # qcut equivalent — 4 balanced groups
        if host_days_active < 2500:
            host_exp = 1
        elif host_days_active < 3800:
            host_exp = 2
        elif host_days_active < 5000:
            host_exp = 3
        else:
            host_exp = 4

        # ── Encoding ────────────────────────────────────
        neighbourhood_enc = neighbourhood_map.get(neighbourhood, 2)
        property_type_enc = property_type_map.get(property_type, 0)
        room_type_enc     = room_type_map.get(room_type, 2)

        # ── Log transform (same as notebook) ────────────
        beds_log = np.log1p(beds)

        # ── EXACT feature order from your notebook ───────
        # ['Neighbourhood', 'Property Type', 'Room Type', 'Beds',
        #  'Review Scores Rating', 'Host Month', 'Host_days_active',
        #  'Host_experience_level', 'reviews_per_day']
        features = np.array([[
            neighbourhood_enc,   # 1. Neighbourhood
            property_type_enc,   # 2. Property Type
            room_type_enc,       # 3. Room Type
            beds_log,            # 4. Beds (log transformed)
            review_score,        # 5. Review Scores Rating
            host_month,          # 6. Host Month
            host_days_active,    # 7. Host_days_active
            host_exp,            # 8. Host_experience_level
            reviews_per_day      # 9. reviews_per_day
        ]])

        if model:
            log_price       = model.predict(features)[0]
            predicted_price = np.expm1(log_price)  # reverse log transform
        else:
            # Demo mode
            base = 80
            if room_type == 'Entire home/apt': base *= 2.5
            elif room_type == 'Private room':  base *= 1.3
            if neighbourhood == 'Manhattan':   base *= 1.8
            elif neighbourhood == 'Brooklyn':  base *= 1.2
            predicted_price = base * (1 + beds * 0.3) * (review_score / 100)

        predicted_price = max(10, round(float(predicted_price), 2))

        return jsonify({
            'success'        : True,
            'predicted_price': predicted_price,
            'price_range'    : {
                'low' : round(predicted_price * 0.85, 2),
                'high': round(predicted_price * 1.15, 2)
            }
        })

    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True, port=5000)