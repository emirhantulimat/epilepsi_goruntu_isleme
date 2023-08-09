import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score




def correlation_info(df):
    correlation_matrix = df.corr()
    correlation_values = correlation_matrix.values
    np.mean(correlation_values)
    plt.figure(figsize=(15, 12))
    sns.heatmap(correlation_matrix, cmap='coolwarm', center=0, annot=False,  linewidths=0.5)
    plt.title('Korelasyon Matrisi')
    plt.show()
    for i in range(len(correlation_values)):
        for j in range(len(correlation_values[i])):
            if correlation_values[i][j] > 0.30:
                print(correlation_values[i][j])

    return correlation_matrix


def dataset_split(df):
    X = df.iloc[:, :-1]  # Son sütunu hariç tüm sütunları alır
    y = df.iloc[:, -1]   # Son sütunu 'Label' bağımlı değişken olarak alır
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print("Eğitim kümesi:")
    print(X_train)
    print(y_train)
    print("\nTest kümesi:")
    print(X_test)
    print(y_test)
    return X_train, X_test, y_train, y_test



df = pd.read_csv(r'features.csv')
df = df.drop("Unnamed: 0", axis=1)

correlation_matrix = correlation_info(df)
X_train, X_test, y_train, y_test = dataset_split(df)

clf = RandomForestClassifier(n_estimators = 100000, max_depth = 45)
model = clf.fit(X_train , y_train)

pre = model.predict(X_test)

conf_matrix = confusion_matrix(y_test, pre)
accuracy = accuracy_score(y_test, pre)













