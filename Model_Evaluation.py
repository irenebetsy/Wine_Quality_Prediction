# Import necessary libraries
from Libraries import np,cycle,pickle,precision_recall_fscore_support,label_binarize,accuracy_score,pd,os,confusion_matrix,plt,sns,roc_curve,auc,roc_auc_score


class ModelEvaluator:
    def __init__(self, model_name, model_file, timestamp):
        self.model_name = model_name
        self.model_file = model_file
        self.timestamp = timestamp
        self.metrics = {}  # Initialize an empty dictionary to store metrics

    def load_model(self):
        with open(self.model_file, "rb") as model_file:
            self.model = pickle.load(model_file)

    def evaluate(self, X_test_data, y_test,evaluation_folder):
        y_pred = self.model.predict(X_test_data)
        # Calculate class-wise precision, recall, F1 score, and support
        classwise_metrics = precision_recall_fscore_support(y_test, y_pred, average=None)
        # Calculate overall accuracy
        overall_accuracy = accuracy_score(y_test, y_pred)


        
        # Save evaluation metrics to a CSV file
        metrics_df = pd.DataFrame({
            "Class": range(len(classwise_metrics[0])),
            "Precision": classwise_metrics[0],
            "Recall": classwise_metrics[1],
            "F1 Score": classwise_metrics[2],
            "Support": classwise_metrics[3]
        })        
        metrics_df.loc[len(classwise_metrics[0])] = ["Accuracy", overall_accuracy, "", "", ""]

        # Save the DataFrame to a CSV file with timestamp in the file name
        output_filename = f'{self.model_name}_Classification_Report_{self.timestamp}.csv'
        #eda_df = pd.DataFrame(eda_results)
        output_path = os.path.join(evaluation_folder,output_filename)
        metrics_df.to_csv(output_path, index=False)



    def confusion_matrix(self, y_test, X_test_data,evaluation_folder):
        y_pred = self.model.predict(X_test_data)
        confusionmatrix = confusion_matrix(y_test, y_pred)


        # Plot and save the confusion matrix
        plt.figure(figsize=(8, 6))
        sns.set(font_scale=1.2)
        sns.heatmap(confusionmatrix, annot=True, fmt="d", cmap="Blues", cbar=False)
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.title(f'Confusion Matrix for {self.model_name}')
        

        confusion_matrix_image_filename = f"{self.model_name}_Confusion_Matrix_{self.timestamp}.png"
        confusion_matrix_image_path = os.path.join(evaluation_folder, confusion_matrix_image_filename)
        plt.savefig(confusion_matrix_image_path)
        plt.close()




    def calculate_roc_auc(self, y_test, X_test_data, evaluation_folder):
        # Binarize the labels for multiclass ROC-AUC calculation
        y_test_bin = label_binarize(y_test, classes=self.model.classes_)
        n_classes = y_test_bin.shape[1]

        y_scores = self.model.predict_proba(X_test_data)

        # Compute ROC curve and ROC area for each class
        fpr = dict()
        tpr = dict()
        roc_auc = dict()

        for i in range(n_classes):
            fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_scores[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])

        # Micro-average ROC curve and ROC area
        fpr["micro"], tpr["micro"], _ = roc_curve(y_test_bin.ravel(), y_scores.ravel())
        roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

        # Macro-average ROC curve and ROC area
        all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
        mean_tpr = np.zeros_like(all_fpr)
        for i in range(n_classes):
            mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])
        mean_tpr /= n_classes
        fpr["macro"] = all_fpr
        tpr["macro"] = mean_tpr
        roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

        # Plot ROC curves
        plt.figure(figsize=(8, 6))
        plt.plot(fpr["micro"], tpr["micro"],
                label=f'Micro-average ROC curve (AUC = {roc_auc["micro"]:0.2f})',
                color='deeppink', linestyle=':', linewidth=4)

        plt.plot(fpr["macro"], tpr["macro"],
                label=f'Macro-average ROC curve (AUC = {roc_auc["macro"]:0.2f})',
                color='navy', linestyle=':', linewidth=4)

        colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
        for i, color in zip(range(n_classes), colors):
            plt.plot(fpr[i], tpr[i], color=color, lw=2,
                    label=f'ROC curve (AUC = {roc_auc[i]:0.2f}) for class {i}')

        plt.plot([0, 1], [0, 1], 'k--', lw=2)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'ROC Curve for {self.model_name}')
        plt.legend(loc='lower right')

        # Save the ROC curve plot
        roc_curve_image_filename = f"{self.model_name}_ROC_Curve_{self.timestamp}.png"
        roc_curve_image_path = os.path.join(evaluation_folder, roc_curve_image_filename)
        plt.savefig(roc_curve_image_path)
        plt.close()



def main(timestamp,desired_output_folder,result_path,model_path):
    transformedScaled_pkl_path=os.path.join(desired_output_folder,f"{timestamp}_transformedScaled_data.pkl")
    with open(transformedScaled_pkl_path, "rb") as file:
        X_train_data,X_test_data,y_train,y_test = pickle.load(file)
    
    # Define the folder for evaluation metrics with timestamp
    evaluation_folder = os.path.join(result_path, f"evaluation_metrics_{timestamp}")
    os.makedirs(evaluation_folder, exist_ok=True)  # Create the folder if it doesn't exist

    nbc_path=os.path.join(model_path,f"{timestamp}_nbc_model.pkl")
    dtc_path=os.path.join(model_path,f"{timestamp}_dtc_model.pkl")
    rfc_path=os.path.join(model_path,f"{timestamp}_rfc_model.pkl")
    knn_path=os.path.join(model_path,f"{timestamp}_knn_model.pkl")
    
    
    # Define the list of trained models and their corresponding filenames
    models = {
        "Naive Bayes Classifier": nbc_path,
        "Decision Tree Classifier": dtc_path,
        "Random Forest Classifier": rfc_path,
        "K Nearest Neighbor": knn_path
    }

    for model_name, model_file in models.items():
        evaluator = ModelEvaluator(model_name, model_file,timestamp)
        evaluator.load_model()
        evaluator.evaluate(X_test_data, y_test,evaluation_folder)
        evaluator.calculate_roc_auc(y_test, X_test_data,evaluation_folder)
        evaluator.confusion_matrix(y_test, X_test_data,evaluation_folder)
    print(f"All evaluations completed. Metrics saved in '{evaluation_folder}' folder.")
