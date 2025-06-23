from Libraries import GridSearchCV,GaussianNB,DecisionTreeClassifier,RandomForestClassifier,KNeighborsClassifier,os,pickle

class BaseClassifier():
    def __init__(self, model):
        self.model = model

    def tune_hyperparameters(self, X_train_data, y_train, param_grid):
        grid_search = GridSearchCV(self.model, param_grid, cv=5)
        grid_search.fit(X_train_data, y_train)
        self.best_model = grid_search.best_estimator_
        self.best_params = grid_search.best_params_

class NB_classifier(BaseClassifier):
    def __init__(self):
        model = GaussianNB()
        super().__init__(model)

class DT_classifier(BaseClassifier):
    def __init__(self):
        model = DecisionTreeClassifier()
        super().__init__(model)

class RF_classifier(BaseClassifier):
    def __init__(self):
        model = RandomForestClassifier()
        super().__init__(model)

class KNN_classifier(BaseClassifier):
    def __init__(self):
        model = KNeighborsClassifier()
        super().__init__(model)





nb_param_grid = {
    'var_smoothing': [1e-9, 1e-8, 1e-7]
}

dtc_param_grid = {
    'max_depth': [None, 5, 10],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

rfc_param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 5, 10],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

knn_param_grid = {
    'n_neighbors': [3, 5, 7],
    'weights': ['uniform', 'distance']
}


class ModelFitting():
    def __init__(self, timestamp, desired_output_folder, model_path):
        transformedScaled_pkl_path = os.path.join(desired_output_folder, f"{timestamp}_transformedScaled_data.pkl")
        with open(transformedScaled_pkl_path, "rb") as file:
            X_train_data, X_test_data, y_train, y_test = pickle.load(file)
            print(X_train_data.columns)

        print("Naive Bayes Classifier")
        nb_classifier = NB_classifier()
        nb_classifier.tune_hyperparameters(X_train_data, y_train, nb_param_grid)
        nbc_model_path = os.path.join(model_path, f"{timestamp}_nbc_model.pkl")
        with open(nbc_model_path, "wb") as model_file:
            pickle.dump(nb_classifier.best_model, model_file)

        print("Decision Tree Classifier")
        dt_classifier = DT_classifier()
        dt_classifier.tune_hyperparameters(X_train_data, y_train,dtc_param_grid)
        dtc_model_path = os.path.join(model_path, f"{timestamp}_dtc_model.pkl")
        with open(dtc_model_path, "wb") as model_file:
            pickle.dump(dt_classifier.best_model, model_file)

        print("Random Forest Classifier")
        rfc_classifier = RF_classifier()
        rfc_classifier.tune_hyperparameters(X_train_data, y_train, rfc_param_grid)
        rfc_model_path = os.path.join(model_path, f"{timestamp}_rfc_model.pkl")
        with open(rfc_model_path, "wb") as model_file:
            pickle.dump(rfc_classifier.best_model, model_file)

        print("K Nearest Neighbor")
        knn_classifier = KNN_classifier()
        knn_classifier.tune_hyperparameters(X_train_data, y_train, knn_param_grid)
        knn_model_path = os.path.join(model_path, f"{timestamp}_knn_model.pkl")
        with open(knn_model_path, "wb") as model_file:
            pickle.dump(knn_classifier.best_model, model_file)
            