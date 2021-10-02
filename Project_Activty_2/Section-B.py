import inline as inline
import matplotlib
import matplotlib_inline
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import sklearn
import warnings
warnings.simplefilter("ignore")

Test = pd.read_csv('F:\Test.csv')
Test.head()
Test.columns
Test.dtypes

print("Test dataset shape", Test.shape)

Test.isnull().sum()

Test["Gender"].fillna(Test["Gender"].mode()[0], inplace=True)
Test["Married"].fillna(Test["Married"].mode()[0], inplace=True)
Test['Dependents'].fillna(Test["Dependents"].mode()[0], inplace=True)
Test["Self_Employed"].fillna(Test["Self_Employed"].mode()[0], inplace=True)
Test["Credit_History"].fillna(Test["Credit_History"].mode()[0], inplace=True)

Test["Loan_Amount_Term"].value_counts()

Test["Loan_Amount_Term"].fillna(Test["Loan_Amount_Term"].mode()[0], inplace=True)

Test["Loan_Amount_Term"].value_counts()

Test["LoanAmount"].fillna(Test["LoanAmount"].median(), inplace=True)

Test.isnull().sum()

Test.isnull().sum()

Test["Gender"].fillna(Test["Gender"].mode()[0], inplace=True)
Test['Dependents'].fillna(Test["Dependents"].mode()[0], inplace=True)
Test["Self_Employed"].fillna(Test["Self_Employed"].mode()[0], inplace=True)
Test["Loan_Amount_Term"].fillna(Test["Loan_Amount_Term"].mode()[0], inplace=True)
Test["Credit_History"].fillna(Test["Credit_History"].mode()[0], inplace=True)
Test["LoanAmount"].fillna(Test["LoanAmount"].median(), inplace=True)

Test.isnull().sum()

Test.info()

# Answer 5 :- Encoding categrical Features:
Test_encoded = pd.get_dummies(Test, drop_first=True)
Test_encoded.head()


#Answer 6:- Split Features and Target Varible
X = Test_encoded.drop(columns='Loan_Status_Y')
y = Test_encoded['Loan_Status_Y']

# Splitting Data

from sklearn.model_selection import Test_test_split

X_Test, X_test, y_Test, y_test = Test_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# Handling/Imputing Missing values

from sklearn.impute import SimpleImputer

imp = SimpleImputer(strategy='mean')
imp_Test = imp.fit(X_Test)
X_Test = imp_Test.transform(X_Test)
X_Test_imp = imp_Test.transform(X_test)

# In[57]:


from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score, f1_score

# In[58]:


tree_clf = DecisionTreeClassifier()
tree_clf.fit(X_Test, y_Test)
y_pred = tree_clf.predict(X_Test)
print("Training Data Set Accuracy: ", accuracy_score(y_Test, y_pred))
print("Training Data F1 Score ", f1_score(y_Test, y_pred))

print("Validation Mean F1 Score: ", cross_val_score(tree_clf, X_Test, y_Test, cv=5, scoring='f1_macro').mean())
print("Validation Mean Accuracy: ", cross_val_score(tree_clf, X_Test, y_Test, cv=5, scoring='accuracy').mean())


training_accuracy = []
val_accuracy = []
training_f1 = []
val_f1 = []
tree_depths = []

for depth in range(1, 20):
    tree_clf = DecisionTreeClassifier(max_depth=depth)
    tree_clf.fit(X_Test, y_Test)
    y_training_pred = tree_clf.predict(X_Test)

    training_acc = accuracy_score(y_Test, y_training_pred)
    train_f1 = f1_score(y_Test, y_training_pred)
    val_mean_f1 = cross_val_score(tree_clf, X_Test, y_Test, cv=5, scoring='f1_macro').mean()
    val_mean_accuracy = cross_val_score(tree_clf, X_Test, y_Test, cv=5, scoring='accuracy').mean()

    training_accuracy.append(training_acc)
    val_accuracy.append(val_mean_accuracy)
    training_f1.append(train_f1)
    val_f1.append(val_mean_f1)
    tree_depths.append(depth)

Tuning_Max_depth = {"Training Accuracy": training_accuracy, "Validation Accuracy": val_accuracy,
                    "Training F1": training_f1, "Validation F1": val_f1, "Max_Depth": tree_depths}
Tuning_Max_depth_df = pd.DataFrame.from_dict(Tuning_Max_depth)

plot_df = Tuning_Max_depth_df.melt('Max_Depth', var_name='Metrics', value_name="Values")
fig, ax = plt.subplots(figsize=(15, 5))
sns.pointplot(x="Max_Depth", y="Values", hue="Metrics", data=plot_df, ax=ax)

import graphviz
from sklearn import tree

tree_clf = tree.DecisionTreeClassifier(max_depth=3)
tree_clf.fit(X_Test, y_Test)
dot_data = tree.export_graphviz(tree_clf, feature_names=X.columns.tolist())
graph = graphviz.Source(dot_data)


training_accuracy = []
val_accuracy = []
training_f1 = []
val_f1 = []
min_samples_leaf = []
import numpy as np

for samples_leaf in range(1, 80, 3):  ### Sweeping from 1% samples to 10% samples per leaf
    tree_clf = DecisionTreeClassifier(max_depth=3, min_samples_leaf=samples_leaf)
    tree_clf.fit(X_Test, y_Test)
    y_training_pred = tree_clf.predict(X_Test)

    training_acc = accuracy_score(y_Test, y_training_pred)
    train_f1 = f1_score(y_Test, y_training_pred)
    val_mean_f1 = cross_val_score(tree_clf, X_Test, y_Test, cv=5, scoring='f1_macro').mean()
    val_mean_accuracy = cross_val_score(tree_clf, X_Test, y_Test, cv=5, scoring='accuracy').mean()

    training_accuracy.append(training_acc)
    val_accuracy.append(val_mean_accuracy)
    training_f1.append(train_f1)
    val_f1.append(val_mean_f1)
    min_samples_leaf.append(samples_leaf)

Tuning_min_samples_leaf = {"Training Accuracy": training_accuracy, "Validation Accuracy": val_accuracy,
                           "Training F1": training_f1, "Validation F1": val_f1, "Min_Samples_leaf": min_samples_leaf}
Tuning_min_samples_leaf_df = pd.DataFrame.from_dict(Tuning_min_samples_leaf)

plot_df = Tuning_min_samples_leaf_df.melt('Min_Samples_leaf', var_name='Metrics', value_name="Values")
fig, ax = plt.subplots(figsize=(15, 5))
sns.pointplot(x="Min_Samples_leaf", y="Values", hue="Metrics", data=plot_df, ax=ax)

# In[64]:


from sklearn.metrics import confusion_matrix

tree_clf = DecisionTreeClassifier(max_depth=3, min_samples_leaf=35)
tree_clf.fit(X_Test, y_Test)
y_pred = tree_clf.predict(X_Test_imp)
print("Test Accuracy: ", accuracy_score(y_test, y_pred))
print("Test F1 Score: ", f1_score(y_test, y_pred))
print("Confusion Matrix on Test Data")
pd.crosstab(y_test, y_pred, rownames=['True'], colnames=['Predicted'], margins=True)

