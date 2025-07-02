#Load the MNIST dataset, train a Random Forest classifier, and compare performance with and without PCA (95% variance).

from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score
import time
import numpy as np


#STep 1 : Load MNIST
print("Loading MNIST...")
X,y = fetch_openml("mnist_784",version=1,return_X_y=True,as_frame=False)
X_train, X_test, y_train, y_test= train_test_split(X,y, train_size=60000, test_size=10000, random_state=42)

#Step 2: Train random forest on orignal data
rf= RandomForestClassifier(n_estimators=100, random_state=42)
start_time= time.time()
rf.fit(X_train, y_train)
train_time_orignal= time.time() - start_time
y_pred= rf.predict(X_test) 
accuracy_orignal= accuracy_score(y_test,y_pred)
print(f"Original Data - Training Time: {train_time_orignal:.2f} seconds")
print(f"Original Data - Test Accuracy: {accuracy_orignal:.4f}")

#Step 3 :Applying pca
pca= PCA(n_components=0.95,random_state=42)
X_train_reduced= pca.fit_transform(X_train)
X_test_reduced= pca.transform(X_test)
print(f"Number of dimensions after PCA: {X_train_reduced.shape[1]}")

#Step 4: Training Random Forest on reudced data
rf_reduced= RandomForestClassifier(n_estimators=100, random_state=42)
start_time= time.time()
rf_reduced.fit(X_train_reduced, y_train)
train_time_reduced= time.time() - start_time
y_pred_reduced =rf_reduced.predict(X_test_reduced)
accuracy_reduced= accuracy_score(y_test, y_pred)
print(f"Reduced Data - Training Time: {train_time_reduced:.2f} seconds")
print(f"Reduced Data - Test Accuracy: {accuracy_reduced:.4f}")

# Step 5: Compare
print("\nComparison:")
print(f"Training Time Reduction: {(train_time_orignal - train_time_reduced) / train_time_orignal * 100:.2f}% faster")
print(f"Accuracy Difference: {accuracy_orignal - accuracy_reduced:.4f}")




