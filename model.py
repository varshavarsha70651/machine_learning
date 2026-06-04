from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix

def train_model(X_train, X_test, y_train, y_test, feature_names):

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)

    # Feature importance (coefficients)
    importance = abs(model.coef_[0])

    return model, accuracy, cm, importance
