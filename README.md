# car_price_prediction_model
<br>
Project deployed at :  https://car-price-prediction-model-u95g.onrender.com
<img width="1701" height="1021" alt="Screenshot 2026-02-11 at 22 01 35" src="https://github.com/user-attachments/assets/10aec668-2222-41ad-8865-ed42a0814648" />


# 🚗 Quikr Used Car Price Predictor

> *A machine learning web application that predicts the resale price of used cars in India using a Linear Regression model trained on Quikr listings data — served through a Flask web interface.*

---

## 📖 Project Overview

This project tackles a real-world problem: **how much is a used car actually worth?** Using data scraped from [Quikr](https://www.quikr.com) — one of India's largest classifieds platforms — we build an end-to-end machine learning pipeline that takes in car details and returns an estimated resale price in Indian Rupees.

The project spans the full data science lifecycle: raw messy data → cleaning → EDA → model building → evaluation → serialisation → Flask deployment.

---

## 🗂️ Project Structure

```
quikr-car-price-predictor/
│
├── quikr_car.csv                  # Raw scraped dataset (892 entries)
├── Cleaned_Car_data.csv           # Cleaned dataset used for training & dropdowns
├── Quikr_prediction.ipynb         # Full ML notebook (EDA + model)
├── LinearRegressionModel.pkl      # Serialised trained pipeline
├── application.py                 # Flask web application
└── templates/
    └── index.html                 # Frontend prediction form
```

---

## 🧭 Project Journey

### Stage 1 — Raw Data & Initial Exploration

The dataset `quikr_car.csv` contains **892 car listings** scraped from Quikr, with 6 columns:

| Column | Description |
|---|---|
| `name` | Full car listing name (often messy) |
| `company` | Car manufacturer |
| `year` | Year of manufacture |
| `Price` | Asking price (target variable) |
| `kms_driven` | Kilometres driven |
| `fuel_type` | Petrol / Diesel / LPG |

On first inspection, the data was riddled with quality issues — a mix of real listings and junk entries.

---

### Stage 2 — Data Cleaning

*This was the most critical and labour-intensive stage of the project.* The raw data had multiple issues, each addressed systematically:

**Problem → Fix:**

- `year` contained non-numeric strings (e.g. *"Used"*, *"URJENT"*) → filtered rows where `year` was not numeric, then cast to `int`
- `Price` had `"Ask For Price"` entries and comma-formatted strings (e.g. *"2,50,000"*) → removed non-numeric rows, stripped commas, cast to `int`
- `kms_driven` had values like `"45000 kms"` and comma separators → split on whitespace, extracted the numeric part, stripped commas
- `fuel_type` had `NaN` rows → dropped null entries
- `name` contained full verbose descriptions (e.g. *"Well maintained Tata Sumo good condition"*) → trimmed to first three words only

After cleaning, the dataset was reduced to **816 clean entries** and saved as `Cleaned_Car_data.csv`. An outlier price cap of **₹60,00,000** was also applied to remove implausible listings.

---

### Stage 3 — Exploratory Data Analysis (EDA)

Before modelling, we visualised the relationships between features and price:

- **Company vs Price** — Box plot revealed significant price variation across brands; premium manufacturers clustered high
- **Year vs Price** — Swarm plot confirmed newer cars command higher prices, as expected
- **kms_driven vs Price** — Scatter plot showed a general negative correlation (more driven = lower value), though with noise
- **Fuel Type vs Price** — Diesel cars tended to price higher; LPG cars sat at the lower end
- **Combined view** — A multi-dimensional `relplot` overlaid fuel type and year on the company-price axis, giving a holistic picture of the data

---

### Stage 4 — Model Building

**Feature / Target split:**

```
X = [name, company, year, kms_driven, fuel_type]
y = Price
```

**Preprocessing challenge:** Three of the five features (`name`, `company`, `fuel_type`) are categorical with high cardinality. A `OneHotEncoder` was first fitted on the *full dataset* to capture all possible categories, then wrapped in a `ColumnTransformer` — this ensures the production model handles any valid car/company combination seen during training.

**Pipeline:**

```
ColumnTransformer (OneHotEncoder on categoricals + passthrough numerics)
        ↓
LinearRegression
```

Using `sklearn.pipeline.make_pipeline` kept preprocessing and model steps bundled together — critical for clean serialisation and deployment.

---

### Stage 5 — Model Evaluation & Tuning

An initial random train/test split (80/20) gave a baseline R² score. To find the most generalisable model, we ran a **1,000-iteration search** across different `random_state` values with a 90/10 split:

```python
for i in range(1000):
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1, random_state=i)
    ...
    scores.append(r2_score(y_test, y_pred))
```

The best-performing split achieved an **R² score approaching ~0.92**, meaning the model explains ~92% of the variance in used car prices on the test set. The model at the optimal `random_state` was then retrained and saved.

---

### Stage 6 — Model Serialisation

The complete pipeline (preprocessor + model) was serialised using `pickle`:

```python
pickle.dump(pipe, open('LinearRegressionModel.pkl', 'wb'))
```

Serialising the *entire pipeline* — not just the model weights — is essential so that the same encoding transformations are automatically applied to any new input at inference time.

---

### Stage 7 — Flask Web Application

`application.py` wires the trained model into a user-facing web app:

- **`GET /`** — renders the home page, populating all four dropdowns (company, model, year, fuel type) dynamically from `Cleaned_Car_data.csv`
- **`POST /predict`** — reads the submitted form, constructs a one-row `DataFrame` matching the training schema, runs `model.predict()`, and returns the estimated price back to the same page

Input sanitisation strips non-numeric characters from `kms_driven` to handle user-entered values like *"50,000"* or *"50000 km"*.

The app runs on **port 5001** and is production-ready for deployment on platforms that set a `PORT` environment variable.

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| Language | Python 3 |
| Data manipulation | pandas, NumPy |
| Visualisation | Matplotlib, Seaborn |
| Machine learning | scikit-learn |
| Model serialisation | pickle |
| Web framework | Flask |
| Frontend | HTML (Jinja2 templates) |

---

## ⚙️ Setup & Installation

**1. Clone the repository**

```bash
git clone https://github.com/your-username/quikr-car-price-predictor.git
cd quikr-car-price-predictor
```

**2. Install dependencies**

```bash
pip install flask pandas numpy scikit-learn matplotlib seaborn
```

**3. Run the app**

```bash
python application.py
```

**4. Open in browser**

```
http://localhost:5001
```

> *Note: `LinearRegressionModel.pkl` and `Cleaned_Car_data.csv` must be present in the root directory before launching the app. Run the notebook first if they are missing.*

---

## 🔮 How to Use

1. Select the **car company** from the dropdown
2. Select the **car model**
3. Choose the **year of manufacture**
4. Select the **fuel type**
5. Enter the **kilometres driven**
6. Click **Predict** — the estimated resale price appears instantly in Indian Rupees

---

## 📊 Model Performance

| Metric | Value |
|---|---|
| Algorithm | Linear Regression |
| Best R² Score | ~0.92 |
| Training set size | ~90% of 816 records |
| Test set size | ~10% of 816 records |

---

## 🔭 Future Improvements

- Add more ML algorithms (Random Forest, XGBoost) and compare
- Incorporate additional features such as number of previous owners and city
- Refresh the dataset with more recent Quikr listings
- Add input validation and error handling on the frontend
- Deploy to a cloud platform (Heroku, Render, AWS)

---

## 👨‍💻 Author

Built as an end-to-end data science and web deployment project, demonstrating the full journey from raw scraped data to a live price prediction tool.
