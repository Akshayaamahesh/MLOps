import joblib
from sklearn import datasets
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split

MODEL_PATH = 'wine_model.joblib'

if __name__ == '__main__':
    # Load the Wine dataset (13 chemical measurements, 3 cultivars)
    wine = datasets.load_wine()
    X, y = wine.data, wine.target
    class_names = [name.replace('_', ' ').title() for name in wine.target_names]

    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)

    # Train a Random Forest (tree-based, so no feature scaling is needed)
    model = RandomForestClassifier(n_estimators=200, max_depth=6, random_state=42)
    model.fit(X_train, y_train)

    # Evaluate on the held-out test set
    y_pred = model.predict(X_test)
    print(f"Test accuracy: {accuracy_score(y_test, y_pred):.3f}")
    print(classification_report(y_test, y_pred, target_names=class_names))

    # Save the model together with the metadata the serving app needs
    joblib.dump({
        'model': model,
        'feature_names': list(wine.feature_names),
        'class_names': class_names,
        'example': X_test[0].tolist(),
    }, MODEL_PATH)
    print(f"Model was trained and saved to {MODEL_PATH}")
