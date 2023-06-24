import h5py
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import seaborn as sns

# HDF5 dosyasını yükleme
raw_data = h5py.File('C:/Users/smndm/Desktop/veri bilimi/Features_Frequency_Alpha_Valence.mat', 'r')
data = raw_data['Features_Frequency_Alpha_Valence']
X_data = data[:-1, :].T  # X'i y'ye şeklini uyduracak şekilde transpoze et
y_data = data[-1, :].flatten()
X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, random_state=42)

accuracy = []
algorithm_names = ['GaussianNB', 'LinearDiscriminantAnalysis', 'SVC', 'KNeighborsClassifier', 'RandomForestClassifier']
#%%
# BayesNet Modeli için eğitim ve sonuç
gnb = GaussianNB()
gnb.fit(X_train, y_train)
y_pred_gnb = gnb.predict(X_test)
accuracy.append(accuracy_score(y_test, y_pred_gnb) * 100)
#%%
# LDA Modeli için eğitim ve sonuç
lda = LinearDiscriminantAnalysis()
lda.fit(X_train, y_train)
y_pred_lda = lda.predict(X_test)
accuracy.append(accuracy_score(y_test, y_pred_lda) * 100)
#%%
# SVM Modeli için eğitim ve sonuç
svc = SVC()
svc.fit(X_train, y_train)
y_pred_svc = svc.predict(X_test)
accuracy.append(accuracy_score(y_test, y_pred_svc) * 100)
#%%
# KNN Modeli için eğitim ve sonuç
knn = KNeighborsClassifier()
knn.fit(X_train, y_train)
y_pred_knn = knn.predict(X_test)
accuracy.append(accuracy_score(y_test, y_pred_knn) * 100)
#%%
# RandomForest Modeli için eğitim ve sonuç
rf = RandomForestClassifier()
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)
accuracy.append(accuracy_score(y_test, y_pred_rf) * 100)
#%%
# Confusion Matrix çizimi
plt.figure(figsize=(12, 8))
for i, alg in enumerate(algorithm_names):
    plt.subplot(2, 3, i + 1)
    model = eval(alg)()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, cmap='Blues', fmt='g')
    plt.title(alg)
    plt.xlabel('Tahmin Etiketleri')
    plt.ylabel('Gerçek Etiketler')

# Accuracy Grafiği çizimi
plt.figure(figsize=(8, 6))
plt.bar(algorithm_names, accuracy, color='blue')
plt.title('Sınıflandırma Algoritmalarının Başarı Oranları')
plt.xlabel('Algoritmalar')
plt.ylabel('Başarı Oranı')
plt.ylim(0, 100)
plt.xticks(rotation=45)
for i in range(len(accuracy)):
   plt.text(algorithm_names[i], accuracy[i], f"{accuracy[i]:.2f}", ha='center', va='bottom')

plt.tight_layout()
plt.show()
#%%