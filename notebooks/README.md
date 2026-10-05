# RE Price Predictor: Notebooks

The analysis behind [RE Price Predictor](https://re-price-predictor.onrender.com), a home valuation model that combines property features with Zillow, Redfin and Realtor.com market signals. The deployed Flask app is in the [parent folder](../).

## Notebooks

Run them in order. They were written in Google Colab against Google Drive.

| Notebook | What it does |
|---|---|
| [`01_Data_Collection.ipynb`](01_Data_Collection.ipynb) | Downloads the five raw sources: Kaggle realtor listings (about 2.2M rows), Zillow home value index and inventory, the Redfin market tracker, and Realtor.com ZIP-level inventory and hotness scores. |
| [`02_Master_Dataset.ipynb`](02_Master_Dataset.ipynb) | Profiles each source, identifies the join keys, cleans each one and merges them into a single master dataset. |
| [`03_RE_Price_Predictor.ipynb`](03_RE_Price_Predictor.ipynb) | Feature engineering, model comparison (Linear Regression, Decision Tree, Random Forest), evaluation, the final sklearn pipeline, and the ZIP market snapshot the app uses at prediction time. |

## Results

On 115,000 held-out sales with complete market data, all scored on the same rows:

| Model | R² (log scale) | Median abs. % error |
|---|---|---|
| Linear Regression | 0.8195 | 17.3% |
| Decision Tree (tuned) | 0.8629 | 13.2% |
| **Random Forest (deployed)** | **0.8743** | **12.3%** |

R² is on log price. In raw dollars on the full held-out set the model scores R² 0.71 with a mean absolute error of about $96.7K. The 115,000 rows are the 71.1% of the test split where every market signal is available. See the notebook for the full comparison and the limitations.

## Data

The raw and processed datasets are not included because of their size and the sources' terms. Re-create them with notebook 01, which documents where each file comes from.

## Environment

Python 3 with pandas, NumPy, scikit-learn, matplotlib and joblib. The notebooks mount Google Drive for storage, so paths will need adjusting to run them elsewhere.
