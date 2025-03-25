import os
import numpy as np
import pandas as pd
import joblib
import torch
from tqdm import tqdm
from transformers import BertTokenizer, BertModel
from sklearn.model_selection import train_test_split, RandomizedSearchCV, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score

# ✅ Create the 'model' directory if it doesn't exist
model_dir = "model"
os.makedirs(model_dir, exist_ok=True)

# ✅ Step 1: Load Dataset
print("📥 Loading dataset... (0%)")
df = pd.read_csv("fake_job_postings.csv")

# ✅ Step 2: Balance the Dataset (5000 Fake & 5000 Real)
df_fake = df[df["fraudulent"] == 1].sample(n=5000, random_state=42)
df_real = df[df["fraudulent"] == 0].sample(n=5000, random_state=42)
df = pd.concat([df_fake, df_real]).sample(frac=1, random_state=42)  # Shuffle
print(f"✅ Balanced Dataset: {df['fraudulent'].value_counts().to_dict()} (10%)")

# ✅ Step 3: Drop Unnecessary Columns & Handle Missing Data
df.drop(columns=['job_id', 'company_profile', 'location'], errors='ignore', inplace=True)
df.fillna("", inplace=True)

if 'description' not in df.columns or 'fraudulent' not in df.columns:
    raise ValueError("Dataset missing required columns: 'description' or 'fraudulent'")
print("✅ Data cleaned! (20%)")

# ✅ Step 4: Encode Categorical Features
categorical_cols = ['title', 'employment_type', 'required_experience', 'required_education', 'industry', 'function']
label_encoders = {}

for col in categorical_cols:
    if col in df.columns:
        df[col] = df[col].astype('category')
        label_encoders[col] = dict(enumerate(df[col].cat.categories))
        df[col] = df[col].cat.codes

print("✅ Categorical features encoded! (30%)")

# ✅ Step 5: Generate BERT Embeddings
print("⏳ Generating BERT embeddings... (40%)")
tqdm.pandas()
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
bert_model = BertModel.from_pretrained("bert-base-uncased").eval()

@torch.no_grad()
def get_bert_embedding(text):
    tokens = tokenizer(text, return_tensors="pt", truncation=True, padding="max_length", max_length=50)
    return bert_model(**tokens).last_hidden_state[:, 0, :].cpu().numpy().flatten()

df['bert_embedding'] = df['description'].progress_apply(get_bert_embedding)
X_text_bert = np.vstack(df['bert_embedding'].values)
print("✅ BERT embeddings generated! (50%)")

# ✅ Step 6: Prepare Numerical Features
X_num = df.drop(columns=['description', 'fraudulent', 'bert_embedding'], errors='ignore')
X_num = X_num.apply(pd.to_numeric, errors='coerce').fillna(0)  # Ensure numeric values

# Combine All Features
X_combined = np.hstack((X_num.values, X_text_bert))
y = df['fraudulent'].values
print("✅ Features combined! (60%)")

# ✅ Step 7: Scale Features
scaler = StandardScaler()
X_combined = scaler.fit_transform(X_combined)
print("✅ Data scaled! (70%)")

# ✅ Step 8: Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X_combined, y, test_size=0.2, random_state=42)
print("✅ Train-test split done! (80%)")

# ✅ Step 9: Train Random Forest with Hyperparameter Tuning
print("🚀 Training Random Forest... (85%)")
param_grid_rf = {'n_estimators': [200, 400], 'max_depth': [15, 25], 'min_samples_split': [2, 3]}
rf = RandomizedSearchCV(RandomForestClassifier(random_state=42), param_grid_rf, n_iter=5, cv=3, n_jobs=-1, verbose=1)
rf.fit(X_train, y_train)
print(f"✅ Best RF Params: {rf.best_params_} (90%)")

# ✅ Step 10: Train XGBoost
print("🚀 Training XGBoost... (92%)")
param_grid_xgb = {
    'n_estimators': [100, 200], 
    'max_depth': [6, 10], 
    'learning_rate': [0.02, 0.05]
}

# Automatically select GPU or CPU for XGBoost
xgb_tree_method = "gpu_hist" if torch.cuda.is_available() else "hist"

xgb_grid = GridSearchCV(
    XGBClassifier(tree_method=xgb_tree_method),
    param_grid_xgb, cv=3, n_jobs=-1, verbose=2
)

# Train GridSearch
xgb_grid.fit(X_train, y_train)
best_xgb_params = xgb_grid.best_params_
print(f"✅ Best XGBoost Params: {best_xgb_params} (93%)")

xgb_final = XGBClassifier(
    **best_xgb_params, 
    tree_method=xgb_tree_method
)
xgb_final.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=0)

print("✅ XGBoost trained! (95%)")

# ✅ Step 11: Train Stacking Model
print("🚀 Training Hybrid Stacking Model... (97%)")
stacking_model = StackingClassifier(
    estimators=[('rf', rf.best_estimator_), ('xgb', xgb_final)], 
    final_estimator=XGBClassifier(tree_method=xgb_tree_method)
)
stacking_model.fit(X_train, y_train)
print("✅ Stacking Model trained! (98%)")

# ✅ Step 12: Evaluate Model
final_preds = stacking_model.predict(X_test)
accuracy = accuracy_score(y_test, final_preds)
print(f"✅ Hybrid Model Accuracy: {accuracy:.4f} (99%)")

# ✅ Step 13: Save Model & Preprocessing Artifacts in 'model' Folder
joblib.dump(stacking_model, os.path.join(model_dir, "fake_job_detection_model.pkl"))
joblib.dump(scaler, os.path.join(model_dir, "scaler.pkl"))
joblib.dump(label_encoders, os.path.join(model_dir, "label_encoders.pkl"))
print(f"🎯 Model & artifacts saved in '{model_dir}' folder! (100%)")
