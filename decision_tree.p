from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from hyperopt import fmin, tpe, hp

# load iris toy dataset
data = load_iris()
X, Y = data['data'], data['target']

# do a random train/test split (used 50% to see stronger gradient in results)
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.50, random_state=42)

# train/test a default Decision Tree on data for initial accuracy
clf = DecisionTreeClassifier()
clf.fit(X_train, y_train)
p_test = clf.predict(X_test)
print('initial_accuracy', accuracy_score(y_test, p_test))

# define an objective function for Hyperopt to optimize
def objective(params):
    # read this iteration's hyperparameter values from dictionary
    max_depth = params['max_depth']
    min_weight_fraction_leaf = params['min_weight_fraction_leaf']
    
    # train/test decision tree on this iteration
    clf = DecisionTreeClassifier(max_depth=max_depth, min_weight_fraction_leaf=min_weight_fraction_leaf)
    clf.fit(X_train, y_train)
    p_test = clf.predict(X_test)
    acc = accuracy_score(y_test, p_test)
    
    # return -accuracy because Hyperopt minimizes objective function
    return -acc

# define grid space of hyperparameters to explore
space = {
    'max_depth':hp.quniform('max_depth', 1, 10, 1),
    'min_weight_fraction_leaf':hp.uniform('min_weight_fraction_leaf', 0, .5),
}

# run Hyperopt - minimizing the objective function, with the given grid space, using TPE method, and 16 max iterations
best = fmin(objective, space, algo=tpe.suggest, max_evals=16)

# train/test decision tree on optimized values
clf = DecisionTreeClassifier(max_depth=best['max_depth'], min_weight_fraction_leaf=best['min_weight_fraction_leaf'])
clf.fit(X_train, y_train)
p_test = clf.predict(X_test)
print('optimized_parms', best)
print('final_accuracy', accuracy_score(y_test, p_test))
