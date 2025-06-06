# Early Stage Parkinson’s Detection Using Voice Analysis

This project uses voice features and machine learning (Random Forest and SVM) to help detect early signs of Parkinson’s Disease. 
Users can input data manually or upload a CSV file to get predictions and visual explanations.


###  Features

-  Predict Parkinson’s using **22 voice measurements**
-  Compare models: **Random Forest vs. SVM**
-  Visual output: **feature importance**, **prediction summary**
-  CSV upload + downloadable results
-  Custom, responsive UI with **Streamlit**
-  Educational tab for learning


### Demo
  
<p align="center">
  <img src="https://github.com/user-attachments/assets/bc49fd6e-f743-4e2b-ab5d-30f31f288582" width="70%">
</p>



<p align="center">
  <img src="https://github.com/user-attachments/assets/c3020847-c9a2-41c3-a807-4483da5b764e" width="70%">
</p>



<p align="center">
   <img src="https://github.com/user-attachments/assets/1b9af9b9-e76d-450a-9b1e-1f5dc5999aa5" width = "70%">
</p>



<p align="center">
   <img src = "https://github.com/user-attachments/assets/af2453ff-56ea-4f80-b0d8-340ea43564d4" width = "70%">
</p>

###  Project Structure

`parkinsons-detection/`

├── `app.py` — Streamlit UI  
├── `model.py` — Random Forest training  
├── `train_svm.py` — SVM model training  
├── `parkinsons_model.pkl` — Trained RF model  
├── `svm_model.pkl` — Trained SVM model  
├── `scaler.pkl` — Feature scaler  
├── `parkinsons.data` — Dataset  
├── `styles.css` — Custom styles for Streamlit  
├── `requirements.txt` — Project dependencies  
└── `README.md` — This file

---

###  Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/parkinsons-detection.git
   cd parkinsons-detection
   ```
2. **Create a virtual environment (optional but recommended)**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Train the models**
   ```bash
   python model.py
   python train_svm.py
   ```

5. **Run the Streamlit app**
   ```bash
   streamlit run app.py
   ```
   
### Model Details

- Random Forest Classifier:

    - Handles non-linear features well
    - Provides feature importance

- SVM (Support Vector Machine):

    - Good for smaller datasets
    - Often better generalization for margin-based classification

Both models use the same scaled feature set derived from the UCI dataset.

### Visualizations
- Feature importance bar chart
- Downloadable CSV with prediction results
- Model comparison toggle (RF vs. SVM)

### Voice Features Used
- MDVP:Fo(Hz), MDVP:Fhi(Hz), MDVP:Flo(Hz)
- Jitter (%, Abs, RAP, PPQ)
- Shimmer (dB, APQ, DDA)
- NHR, HNR
- RPDE, DFA
- Spread1, Spread2
- D2, PPE
  
Total: 22 features.Voice Features Used



*The earlier the diagnosis, the better the chance to manage symptoms effectively.*

### License
This project is licensed under the MIT License.


