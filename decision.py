# Importing Libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
import pickle
import warnings
warnings.filterwarnings('ignore')

# Loading Dataset
data = pd.read_csv("https://raw.githubusercontent.com/Dhavaltharkar/Breast_Cancer_Detection/main/breast_cancer.csv")

# Defining X and y variables and assigning the values from the dataset
X = data.iloc[:, 1:-1].values
y = data.iloc[:, -1].values

# Splitting the dataset into the Training set and Test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Decision Tree
dec_model = DecisionTreeClassifier()
dec_model.fit(X_train, y_train)

# Saving model to disk
pickle.dump(dec_model, open('dec_model.pkl', 'wb'))
