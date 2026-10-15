import os
import pickle
import numpy as np
from flask import Flask, request, jsonify, render_template_string

app = Flask(__name__)

# Load the trained linear regression model
MODEL_PATH = os.path.join(os.path.dirname(__file__), "linear.pkl")
with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)

# Expected features from linear.pkl
FEATURE_NAMES = [
    "Square_Footage",
    "Num_Bedrooms",
    "Num_Bathrooms",
    "Year_Built",
    "Lot_Size",
    "Garage_Size",
    "Neighborhood_Quality"
]

HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>House Price Predictor</title>
  <style>
    :root {
      --bg: #0f172a;
      --card-bg: rgba(30, 41, 59, 0.7);
      --border: rgba(255, 255, 255, 0.1);
      --accent: #6366f1;
      --accent-hover: #4f46e5;
      --text: #f8fafc;
      --muted: #94a3b8;
    }

    * {
      box-sizing: border-box;
      margin: 0;
      padding: 0;
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Oxygen, Ubuntu, Cantarell, sans-serif;
    }

    body {
      background: radial-gradient(circle at 15% 20%, #1e1b4b 0%, var(--bg) 60%);
      color: var(--text);
      min-height: 100vh;
      display: flex;
      justify-content: center;
      align-items: center;
      padding: 2rem 1rem;
    }

    .container {
      background: var(--card-bg);
      backdrop-filter: blur(14px);
      -webkit-backdrop-filter: blur(14px);
      border: 1px solid var(--border);
      border-radius: 20px;
      padding: 2.5rem;
      width: 100%;
      max-width: 680px;
      box-shadow: 0 25px 50px -12px rgba(0, 0, 0, 0.45);
    }

    h1 {
      font-size: 1.85rem;
      font-weight: 700;
      margin-bottom: 0.5rem;
      text-align: center;
    }

    p.subtitle {
      color: var(--muted);
      font-size: 0.95rem;
      text-align: center;
      margin-bottom: 2rem;
    }

    .grid {
      display: grid;
      grid-template-columns: repeat(2, 1fr);
      gap: 1.25rem;
    }

    .full-width {
      grid-column: span 2;
    }

    .form-group {
      display: flex;
      flex-direction: column;
      gap: 0.4rem;
    }

    label {
      font-size: 0.85rem;
      font-weight: 600;
      color: var(--muted);
      text-transform: uppercase;
      letter-spacing: 0.05em;
    }

    input {
      background: rgba(15, 23, 42, 0.6);
      border: 1px solid var(--border);
      color: #fff;
      padding: 0.75rem 1rem;
      border-radius: 10px;
      font-size: 1rem;
      transition: border-color 0.2s ease, box-shadow 0.2s ease;
    }

    input:focus {
      outline: none;
      border-color: var(--accent);
      box-shadow: 0 0 0 3px rgba(99, 102, 241, 0.25);
    }

    button {
      margin-top: 1.5rem;
      background: var(--accent);
      color: #fff;
      border: none;
      border-radius: 10px;
      padding: 0.9rem;
      font-size: 1rem;
      font-weight: 600;
      cursor: pointer;
      width: 100%;
      transition: all 0.2s ease;
      display: flex;
      justify-content: center;
      align-items: center;
      gap: 0.5rem;
    }

    button:hover {
      background: var(--accent-hover);
      transform: translateY(-1px);
    }

    button:active {
      transform: translateY(0);
    }

    .result-box {
      margin-top: 1.75rem;
      padding: 1.25rem;
      border-radius: 12px;
      background: rgba(99, 102, 241, 0.1);
      border: 1px solid rgba(99, 102, 241, 0.3);
      text-align: center;
      display: none;
    }

    .result-label {
      font-size: 0.85rem;
      color: var(--muted);
      text-transform: uppercase;
      letter-spacing: 0.05em;
    }

    .result-val {
      font-size: 2rem;
      font-weight: 700;
      color: #818cf8;
      margin-top: 0.25rem;
    }

    @media (max-width: 580px) {
      .grid {
        grid-template-columns: 1fr;
      }
      .full-width {
        grid-column: span 1;
      }
    }
  </style>
</head>
<body>
  <div class="container">
    <h1>House Valuation</h1>
    <p class="subtitle">Linear regression estimator based on structural features</p>

    <form id="prediction-form">
      <div class="grid">
        <div class="form-group">
          <label for="Square_Footage">Square Footage</label>
          <input type="number" id="Square_Footage" name="Square_Footage" step="any" placeholder="e.g. 2100" required>
        </div>

        <div class="form-group">
          <label for="Num_Bedrooms">Bedrooms</label>
          <input type="number" id="Num_Bedrooms" name="Num_Bedrooms" step="1" placeholder="e.g. 3" required>
        </div>

        <div class="form-group">
          <label for="Num_Bathrooms">Bathrooms</label>
          <input type="number" id="Num_Bathrooms" name="Num_Bathrooms" step="0.5" placeholder="e.g. 2.5" required>
        </div>

        <div class="form-group">
          <label for="Year_Built">Year Built</label>
          <input type="number" id="Year_Built" name="Year_Built" step="1" placeholder="e.g. 2005" required>
        </div>

        <div class="form-group">
          <label for="Lot_Size">Lot Size (sq ft)</label>
          <input type="number" id="Lot_Size" name="Lot_Size" step="any" placeholder="e.g. 5000" required>
        </div>

        <div class="form-group">
          <label for="Garage_Size">Garage Capacity (cars)</label>
          <input type="number" id="Garage_Size" name="Garage_Size" step="1" placeholder="e.g. 2" required>
        </div>

        <div class="form-group full-width">
          <label for="Neighborhood_Quality">Neighborhood Quality (1 - 10)</label>
          <input type="number" id="Neighborhood_Quality" name="Neighborhood_Quality" min="1" max="10" step="any" placeholder="e.g. 7" required>
        </div>
      </div>

      <button type="submit" id="submit-btn">Run Estimation</button>
    </form>

    <div class="result-box" id="result-container">
      <div class="result-label">Estimated Market Value</div>
      <div class="result-val" id="output-val">$0.00</div>
    </div>
  </div>

  <script>
    const form = document.getElementById("prediction-form");
    const resultBox = document.getElementById("result-container");
    const outputVal = document.getElementById("output-val");
    const submitBtn = document.getElementById("submit-btn");

    form.addEventListener("submit", async (e) => {
      e.preventDefault();
      submitBtn.innerText = "Calculating...";
      submitBtn.disabled = true;

      const formData = new FormData(form);
      const payload = {};
      formData.forEach((value, key) => {
        payload[key] = parseFloat(value);
      });

      try {
        const response = await fetch("/predict", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(payload)
        });

        const data = await response.json();

        if (response.ok) {
          resultBox.style.display = "block";
          outputVal.innerText = new Intl.NumberFormat('en-US', {
            style: 'currency',
            currency: 'USD'
          }).format(data.prediction);
        } else {
          alert("Error: " + (data.error || "Prediction request failed"));
        }
      } catch (err) {
        alert("Failed to communicate with prediction service.");
      } finally {
        submitBtn.innerText = "Run Estimation";
        submitBtn.disabled = false;
      }
    });
  </script>
</body>
</html>
"""

@app.route("/", methods=["GET"])
def index():
    return render_template_string(HTML_TEMPLATE)

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json(force=True)
        # Extract features in the precise order expected by the model
        features = [float(data[feature]) for feature in FEATURE_NAMES]
        
        # Reshape to (1, n_features)
        feature_array = np.array(features).reshape(1, -1)
        
        prediction = model.predict(feature_array)[0]
        return jsonify({"prediction": float(prediction)})
    except KeyError as e:
        return jsonify({"error": f"Missing feature: {str(e)}"}), 400
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
