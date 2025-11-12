"""
Compare Multiple Models on Extracted Features (No Cross-Validation)
==================================================================
Trains and evaluates each classifier on train/test split only.
"""

import numpy as np
import pandas as pd
import os
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, classification_report
import xgboost as xgb
import time

# Load features (supports .npz and .pkl)
def load_features(features_dir='./features', model_name='resnet50'):
    npz_locations = [
        f'./data/cifar10_{model_name}_features.npz',
        f'cifar10_{model_name}_features.npz',
        f'./cifar10_{model_name}_features.npz'
    ]
    for npz_file in npz_locations:
        if os.path.exists(npz_file):
            print(f"\n[INFO] Loading NPZ features: {npz_file}")
            data = np.load(npz_file)
            X_train = data['train_features']
            y_train = data['train_labels']
            X_test = data['test_features']
            y_test = data['test_labels']
            return X_train, X_test, y_train, y_test
    train_file = os.path.join(features_dir, f'cifar10_train_features_{model_name}.pkl')
    test_file = os.path.join(features_dir, f'cifar10_test_features_{model_name}.pkl')
    with open(train_file, 'rb') as f:
        train_data = pickle.load(f)
    with open(test_file, 'rb') as f:
        test_data = pickle.load(f)
    X_train = train_data['features']
    y_train = train_data['labels']
    X_test = test_data['features']
    y_test = test_data['labels']
    return X_train, X_test, y_train, y_test

# Define models to compare
def get_models():
    models = {
        'Random Forest': RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1),
        'XGBoost': xgb.XGBClassifier(n_estimators=100, eval_metric='mlogloss', random_state=42, n_jobs=-1),
        'Logistic Regression': LogisticRegression(max_iter=200, random_state=42, n_jobs=-1),
        'KNN': KNeighborsClassifier(n_neighbors=5, n_jobs=-1),
    }
    return models

# Main comparison loop
def main():
    print("=" * 80)
    print("MODEL COMPARISON ON EXTRACTED FEATURES (NO CV)")
    print("=" * 80)
    X_train, X_test, y_train, y_test = load_features()
    print(f"Full dataset - Training: {X_train.shape}, Test: {X_test.shape}")
    
    # Flatten labels if needed
    y_train = y_train.flatten()
    y_test = y_test.flatten()
    
    # Use stratified subset to maintain class balance
    from sklearn.model_selection import train_test_split
    subset_train = 5000  # 500 samples per class
    subset_test = 1000   # 100 samples per class
    
    # Stratified sampling to keep class distribution
    X_train, _, y_train, _ = train_test_split(
        X_train, y_train, train_size=subset_train, stratify=y_train, random_state=42
    )
    X_test, _, y_test, _ = train_test_split(
        X_test, y_test, train_size=subset_test, stratify=y_test, random_state=42
    )
    
    print(f"Using stratified subset - Training: {X_train.shape}, Test: {X_test.shape}")
    print(f"Train class distribution: {np.bincount(y_train)}")
    print(f"Test class distribution: {np.bincount(y_test)}")
    
    # Standardize features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    models = get_models()
    results = []
    for name, model in models.items():
        print(f"\nTraining {name}...")
        start = time.time()
        model.fit(X_train_scaled, y_train)
        train_acc = accuracy_score(y_train, model.predict(X_train_scaled))
        test_pred = model.predict(X_test_scaled)
        test_acc = accuracy_score(y_test, test_pred)
        test_f1 = f1_score(y_test, test_pred, average='weighted')
        elapsed = time.time() - start
        print(f"  Train acc: {train_acc:.4f} | Test acc: {test_acc:.4f} | Test F1: {test_f1:.4f} | Time: {elapsed:.1f}s")
        results.append({
            'Model': name,
            'Train Accuracy': train_acc,
            'Test Accuracy': test_acc,
            'Test F1': test_f1,
            'Time (s)': elapsed
        })
    # Summary table
    df = pd.DataFrame(results)
    print("\n" + "=" * 80)
    print("SUMMARY TABLE")
    print("=" * 80)
    print(df.sort_values('Test Accuracy', ascending=False).to_string(index=False))
    print("\n" + "=" * 80)
    print("Best model for DIO optimization: ")
    best = df.sort_values('Test Accuracy', ascending=False).iloc[0]
    print(f"  {best['Model']} (Test Accuracy: {best['Test Accuracy']:.4f})")

    # Visualization
    import matplotlib.pyplot as plt
    plt.figure(figsize=(10, 5))
    ax = df.sort_values('Test Accuracy', ascending=False).plot(
        x='Model', y='Test Accuracy', kind='bar', legend=False, color='skyblue',
        title='Test Accuracy by Model', ylim=(0, 1)
    )
    plt.ylabel('Test Accuracy')
    plt.xlabel('Model')
    plt.xticks(rotation=30, ha='right')
    plt.tight_layout()
    plt.savefig('model_test_accuracy.png', dpi=120)
    plt.show()

    plt.figure(figsize=(10, 5))
    ax2 = df.sort_values('Test F1', ascending=False).plot(
        x='Model', y='Test F1', kind='bar', legend=False, color='lightgreen',
        title='Test F1-score by Model', ylim=(0, 1)
    )
    plt.ylabel('Test F1-score')
    plt.xlabel('Model')
    plt.xticks(rotation=30, ha='right')
    plt.tight_layout()
    plt.savefig('model_test_f1.png', dpi=120)
    plt.show()

if __name__ == "__main__":
    main()
