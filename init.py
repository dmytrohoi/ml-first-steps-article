import pandas
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.externals import joblib
from sklearn.datasets import load_iris

# Завантаження датасету
URL = "iris.csv" #https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
NAMES = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']
iris = pandas.read_csv(URL, names=NAMES)

# Перевірка кількості значень в сеті
print("Значень в сеті: {}".format(iris.shape))

# Розмежування target функцій від вхідних даних
y = iris['class']
X = iris.drop('class',axis=1)

# Використовуючи вбудовану функцію поділимо дані на тестові й перевірочні
X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                    test_size=0.3, # % кількість тестових значень
                                                    random_state=50, 
                                                    stratify=y) # значення для перевірки правильності

# Побудова дерева рішень на базі наших значень
iris_tree = tree.DecisionTreeClassifier()
iris_tree.fit(X_train,y_train)

# Ця функція scipy має наступні параметри
#
#DecisionTreeClassifier(class_weight=None, criterion='gini', max_depth=None,
#            max_features=None, max_leaf_nodes=None,
#            min_impurity_split=1e-07, min_samples_leaf=1,
#            min_samples_split=2, min_weight_fraction_leaf=0.0,
#            presort=False, random_state=None, splitter='best')

# Для візуалізації дерева використаємо вбудовану функцію
iris=load_iris()
print(tree.export_graphviz(iris_tree, # Дерево рішень натреноване
                         out_file='iris.dot', # Вихідне зображення
                         feature_names=iris.feature_names,
                         class_names=iris.target_names,  
                         filled=True, rounded=True,
                         special_characters=True))

# Виконаємо перевірку моделі взявши значення з тестового сету
y_pred = (iris_tree.predict(X_test))
print ("Перевірка на 30% даних. Точність передбачення: {}".format(accuracy_score(y_test, y_pred)* 100));

# Збереження моделі дерева
joblib.dump(iris_tree, 'iris.pkl')

# Відновлення моделі з файлу
# iris_tree_backup = joblib.load('iris.pkl')
# iris_tree_backup.predict(X_test)