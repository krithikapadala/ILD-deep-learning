from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

def evaluate_model(y_true, y_pred):
    accuracy = accuracy_score(y_true, y_pred)
    matrix = confusion_matrix(y_true, y_pred)

    print("Accuracy:", accuracy)
    print("Confusion Matrix:\n", matrix)
    print("Classification Report:\n", classification_report(y_true, y_pred))
