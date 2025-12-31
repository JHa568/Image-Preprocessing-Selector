from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, accuracy_score
from skimage import io 

import joblib
import cv2
import numpy as np


image_loc = "../images"

image = cv2.imread(image_loc+'/4.png')

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
gray = cv2.resize(gray, (28, 28))
gray_np = np.asarray(gray)

digits = datasets.load_digits()

X = digits.data   # Flattened pixel values
y = digits.target # Labels 0-9

# print(X[0], len(X[0]))

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.4, random_state=42
)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
# print(X_train[0], len(X_train[0]))
knn = KNeighborsClassifier(n_neighbors=3)  # k=3
knn.fit(X_train, y_train)

y_pred = knn.predict(X_test)

# 6. Evaluate
# print("Accuracy:", accuracy_score(y_test, y_pred))
# print("\nClassification Report:\n", classification_report(y_test, y_pred))
# g_flat = [gray_np.flatten()]
# final_test = knn.predict(g_flat)
#print(final_test)
if accuracy_score(y_test, y_pred) > 0.9:
    joblib.dump(knn, "digit_classifier.pkl")
    joblib.dump(scaler, "digit_scaler.pkl")
    
cv2.imshow("gray", gray)
cv2.waitKey(0)
cv2.destroyAllWindows()
    