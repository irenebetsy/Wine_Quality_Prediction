from Libraries import StandardScaler,PCA,pickle,pd,os

#from data import y
class preprocessing_class:
    def __init__(self,cols_names):
        self.cols_names=cols_names

    def cols_names_func(self):
        self.num_cols=self.cols_names.num_cols
        self.cat_cols=self.cols_names.cat_cols
        self.num_cols_cols=self.cols_names.num_cols_cols
        self.cat_cols_cols=self.cols_names.cat_cols_cols 
        
class Transformation:
    def __init__(self,preprocessing_class,timestamp):
        self.num_cols=preprocessing_class.num_cols
        self.cat_cols=preprocessing_class.cat_cols
        self.num_cols_cols=preprocessing_class.num_cols_cols
        self.cat_cols_cols=preprocessing_class.cat_cols_cols
        self.timestamp=timestamp

        #for col in self.data.columns:
        #    if col not in self.num_cols:

    def standard_scaler(self,desired_output_folder):
        data_split_pkl_path=os.path.join(desired_output_folder,f"{self.timestamp}_data_split.pkl")
        with open(data_split_pkl_path, "rb") as file:
            X_train,X_test, y_train,y_test = pickle.load(file)
        X_train_scaled_list=[]
        X_test_scaled_list=[]
        for col in X_train.columns:
            if col not in self.cat_cols:
                scaler = StandardScaler()
                X_train_scaled_data=scaler.fit_transform(X_train[[col]])
                X_test_scaled_data=scaler.transform(X_test[[col]])
                df_X_train_scaled=pd.DataFrame(X_train_scaled_data,columns=[col])
                X_train_scaled_list.append(df_X_train_scaled)
                df_X_test_scaled=pd.DataFrame(X_test_scaled_data,columns=[col])
                X_test_scaled_list.append(df_X_test_scaled)
        self.X_train_scaled=pd.concat(X_train_scaled_list,axis=1)
        self.X_test_scaled=pd.concat(X_test_scaled_list,axis=1)
        # Inside transformation.py, after fitting the scaler
        scaler_path=os.path.join(desired_output_folder,f"{self.timestamp}_scaler.pkl")
        with open(scaler_path, "wb") as scaler_file:
            pickle.dump((scaler), scaler_file)
        print(scaler.mean_)
        print(scaler.scale_)
        return self.X_train_scaled,self.X_test_scaled
    
    def concat_scaled_cat(self,desired_output_folder):
        data_split_pkl_path=os.path.join(desired_output_folder,f"{self.timestamp}_data_split.pkl")
        with open(data_split_pkl_path, "rb") as file:
            X_train,X_test, y_train,y_test = pickle.load(file)
        cat_list_train=[]
        cat_list_test=[]
        for col in X_train.columns:
            if col not in self.num_cols: 
                cat_list_train.append(X_train[[col]])
        self.df_cat_list_train=pd.concat(cat_list_train,axis=1)
        for col in X_test.columns:
            if col not in self.num_cols:
                cat_list_test.append(X_test[[col]])
        self.df_cat_list_test=pd.concat(cat_list_test,axis=1)
        # Reset the index of the dataframes
        self.X_train_scaled.reset_index(drop=True, inplace=True)
        self.df_cat_list_train.reset_index(drop=True, inplace=True)
        self.X_train_data=pd.concat([self.X_train_scaled,self.df_cat_list_train],axis=1)  
        self.X_test_scaled.reset_index(drop=True, inplace=True)
        self.df_cat_list_test.reset_index(drop=True, inplace=True)   
        self.X_test_data=pd.concat([self.X_test_scaled,self.df_cat_list_test],axis=1)      
        X_train_data=self.X_train_data
        X_test_data=self.X_test_data
        ScaledData_path=os.path.join(desired_output_folder,f"{self.timestamp}_transformedScaled_data.pkl")
        with open(ScaledData_path, "wb") as file:
            pickle.dump((X_train_data,X_test_data,y_train,y_test), file)
        return X_train_data,X_test_data

    def apply_pca(self,n_components=3):
        pca = PCA(n_components=n_components)
        x_train_pca=pca.fit_transform(self.X_train_data)
        x_test_pca=pca.transform(self.X_test_data)
        #print(x_train_pca)
        return x_train_pca,x_test_pca

    

"""if __name__ == "__main__":

    dataset_name=Adr_file_path
    time_stamp=datetime.now().strftime("%Y%m%d_%H%M%S")

    # timestamp
    pkl_files=[f for f in os.listdir(data_prep_output_folder)if f.endswith(".pkl")]
    latest_pkl_file=max(pkl_files,key=lambda x:os.path.getmtime(os.path.join(data_prep_output_folder,x)))
    timestamp_new=latest_pkl_file.split("_data_split.pkl")[0]
    data_split_pkl_path=os.path.join(data_prep_output_folder,f"{timestamp_new}_data_split.pkl")


    prep=preprocessing(dataset_name)
    prepclass=preprocessing_class(prep)
    prepclass.cols_names_func()

    transformation = Transformation(prepclass,time_stamp)
    transformation.standard_scaler()
    transformation.concat_scaled_cat()

    # Perform PCA
    X_train_pca,X_test_pca= transformation.apply_pca()
    with open("transformedPCA_data.pkl", "wb") as file:
        pickle.dump((X_train_pca,X_test_pca,y_train,y_test), file)"""
