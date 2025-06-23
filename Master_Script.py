# Import necessary modules and classes
from Data import data_file_path,data_prep_pkl_path,data_model_path,data_result_path,y
from Libraries import datetime,pd
from Preprocessing import preprocessing, impute_encode
from Transformation import preprocessing_class, Transformation
from Model_Fitting import ModelFitting
from Model_Evaluation import main

desired_output_folder=data_prep_pkl_path
model_path=data_model_path
result_path=data_result_path
#
#from model_fitting import ML_regressor, BL_regressor, DT_regressor, mlr_param_grid, blr_param_grid, dtr_param_grid
#from model_evaluation import ModelEvaluator, main

data=pd.read_csv(data_file_path, encoding='latin-1')
# Drop specified columns
#data = data.drop(['DOJ','DOR','Emp_ID', 'Emp_Name','Emp_Status','TenureinDays','Target_Cat'], axis=1)

dataset_name="data"
timestamp=datetime.now().strftime("%Y%m%d_%H%M%S")

# Preprocessing.py
pre_processing = preprocessing(data)
imputeencode=impute_encode(pre_processing,timestamp)
imputeencode.perform_imputation(desired_output_folder)
imputeencode.winsorization(y,desired_output_folder)
imputeencode.perform_encoding(y,desired_output_folder)
imputeencode.split_data(desired_output_folder,y)

#Transformation
prep=preprocessing(data)
prepclass=preprocessing_class(prep)
prepclass.cols_names_func()
transformation = Transformation(prepclass,timestamp)
transformation.standard_scaler(desired_output_folder)
transformation.concat_scaled_cat(desired_output_folder)

#ModelFitting
modelfitting=ModelFitting(timestamp,desired_output_folder,model_path)

#ModelEvaluation
modelevaluation=main(timestamp,desired_output_folder,result_path,model_path)
