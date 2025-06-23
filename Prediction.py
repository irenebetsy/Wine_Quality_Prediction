from Libraries import pd, pickle, np, os
from datetime import datetime

data_model_folder = r"SQL_Data\Model"
data_model_path = data_model_folder.replace('\\', '/')
data_prep_pkl = r"SQL_Data\Data_Prep_pkl"
data_prep_pkl_path = data_prep_pkl.replace('\\', '/')
data_result_folder = r"SQL_Data\Result"
data_result_path = data_result_folder.replace('\\', '/')


class Wine_Quality:

    def callPredict(self,input_data):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Load input data
        data_path = r"SQL_Data\Input\Test_Data.csv"
        input_data_path = data_path.replace('\\', '/')
        input_data = pd.read_csv(input_data_path, encoding='ISO-8859-1')

        orig_data = input_data.copy()

        # Load encoding mapping
        encode_path = os.path.join(data_prep_pkl_path, "20250621_114102_encode_mapping.pkl")
        with open(encode_path, "rb") as file:
            label_mapping = pickle.load(file)

        # Load scaler
        scaler_path = os.path.join(data_prep_pkl_path, "20250621_114102_scaler.pkl")
        with open(scaler_path, "rb") as f:
            scaler = pickle.load(f)

        # Load model
        model_file_path = os.path.join(data_model_path, "20250621_114102_dtc_model.pkl")
        with open(model_file_path, 'rb') as model_file:
            loaded_model = pickle.load(model_file)

        # Encode categorical columns
        for col in input_data.select_dtypes(include=['object']).columns:
            if col in label_mapping:
                input_data[col] = input_data[col].map(label_mapping[col])
            input_data[col] = input_data[col].fillna(-1)

        # Separate numeric and categorical again (after encoding)
        num_cols = input_data.select_dtypes(include=[np.number]).columns.tolist()
        cat_cols = input_data.columns.difference(num_cols).tolist()

        # Scale numeric columns (all at once)
        scaled_array = scaler.fit_transform(input_data[num_cols])
        X_Scaled_data = pd.DataFrame(scaled_array, columns=num_cols)

        # Combine with categorical data
        data = pd.concat([X_Scaled_data, input_data[cat_cols].reset_index(drop=True)], axis=1)


        # Optional debug output
        print("Final data shape before prediction:", data.shape)

        # Predict
        predict_result = loaded_model.predict(data)

        """# Append prediction to original input
        orig_data['Predicted'] = predict_result"""

        # Map predictions to human-readable labels
        prediction_labels = {0: "Poor", 1: "Good", 2: "Vintage"}
        #orig_data['Predicted'] = [prediction_labels[p] for p in predict_result]
        orig_data['Predicted'] = [prediction_labels.get(p, "Unknown") for p in predict_result]

        # Save prediction results
        output_file = os.path.join(data_result_path, f"{timestamp}_predicted_data.csv")
        orig_data.to_csv(output_file, index=False)
        print(f" Prediction saved to: {output_file}")



        return orig_data.to_json(orient='records')
"""if __name__ == "__main__":
    attrition_result = Wine_Quality()
    attrition_result.callPredict()"""
       