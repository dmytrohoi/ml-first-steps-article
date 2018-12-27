import pandas
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.externals import joblib

# Завантаження датасету
URL = "iris.csv" # "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
NAMES = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']
iris = pandas.read_csv(URL, names=NAMES)

# Перевірка кількості значень в сеті
print("Значень в сеті: {s[0]}, рядків: {s[1]}".format(s=iris.shape))

# Розмежування target функцій від вхідних даних
y = iris['class']
X = iris.drop('class',axis=1)

# Відновлення моделі з файлу
try:
    iris_tree_backup = joblib.load('iris.pkl')
finally:
    print("Модель завантажено.\n")

# Побудова передбачення для 100% даних
y_pred = iris_tree_backup.predict(X)
print ("Перевірка на 100% даних. Точність передбачення: {}".format( accuracy_score(y, y_pred) * 100 ))

while True:
    tests = input("\nСпробувати ще раз? (Y/n) - ")

    if tests == "Y":
        pass
    elif tests == "n":
        break
    else:
        print("[Помилка] Неправильне введення.\n\n")
        break
    
    while True:
        test_size = int(input("\nКількість даних у відсотках: "))
        if test_size > 0 and test_size < 101:
            test_size = test_size / 100
            break
        else:
            print("[Помилка]Будь ласка введіть значення від 1% до 100% без знаку '%'\n")

    X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                   test_size=test_size, # % кількість тестових значень
                                                   random_state=100, 
                                                   stratify=y) # значення для перевірки правильності
    y_pred = iris_tree_backup.predict(X_test)
    print ("Перевірка на {}% даних. Точність передбачення: {}%".format(test_size * 100, accuracy_score(y_test, y_pred) * 100 ))

