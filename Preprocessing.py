from Libraries import pd,np,train_test_split,pickle,LabelEncoder,os,plt,sns

class preprocessing:
    def __init__(self,data):
        self.data = data
        self.dataset_name=self.data
        self.num_cols = self.data.select_dtypes(include=[np.number])
        self.cat_cols = self.data.select_dtypes(include=['object'])
        self.num_cols_cols=self.num_cols.columns
        self.cat_cols_cols=self.cat_cols.columns

class impute_encode:
    def __init__(self,preprocessing,timestamp):
        self.num_cols=preprocessing.num_cols
        self.cat_cols=preprocessing.cat_cols
        self.data=preprocessing.data
        self.timestamp=timestamp
        
    def perform_imputation(self,desired_output_folder):
        self.data.dropna(axis=1,how='all',inplace=True)
        na=self.data.isnull().sum()
        print(na)
        na_sum=sum(na)
        na_max = self.data.isnull().sum().max()
        print(na_max)
        if na_max<=10:
            data=self.data.dropna(inplace=True) 
        else:
            for numcol in self.num_cols:                 
                mean_value = self.num_cols.mean()
                data_num = self.num_cols.fillna(mean_value)
                print("Mean value used to fill null values:", mean_value)
            for catcol in self.cat_cols:
                mode_value = self.cat_cols.mode().iloc[0]
                data_cat = self.cat_cols.fillna(mode_value)
                print("Mode value used to fill null values:", mode_value)
    
            values_to_replace = {
                'mean_value': mean_value,
                'mode_value': mode_value
            }
            print(values_to_replace)
            # Store the dictionary in a PKL file
            nv_filename = self.timestamp + "_null_values.pkl"
            null_values_file_path = os.path.join(desired_output_folder, f"{nv_filename}")
            with open(null_values_file_path, 'wb') as f:
                pickle.dump((mean_value,mode_value), f)
            print("Mean value and Mode value have been stored in values.pkl.")
                
            merged_data = pd.concat([data_cat, data_num], axis=1)
            merged_data .tail()     
            self.data=merged_data
        self.data=self.data.drop_duplicates()
        print(self.data.isnull().sum())
        print(self.data.shape)
        print(self.data)
        return self.data

    """#Treating outliers (Winsorization)
    def winsorization(self,desired_output_folder):
        # Box plot before winsorization
        plt.figure(figsize=(12, 8))
        sns.boxplot(data=self.data[self.num_cols.columns])
        plt.title("Box Plot before Winsorization")
        plt.xticks(rotation=45)
        # Adjust layout to prevent x-axis label overlapping
        plt.tight_layout()
        boxplot_folder=f'Box_Plots_{self.timestamp}'
        output_path = os.path.join(desired_output_folder,boxplot_folder,f'_BoxPlot_before_Winsorization_{self.timestamp}.png')
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path)
        plt.close()
        plt.close()
        plt.show()
        print("******") 

        for col in self.num_cols.columns:
            # Calculate the first and third quartiles
            Q1 = self.num_cols[col].quantile(0.25)
            Q3 = self.num_cols[col].quantile(0.75)
            # Set boundaries
            lower_bound = Q1 - 1.5 * (Q3 - Q1)
            upper_bound = Q3 + 1.5 * (Q3 - Q1)
            # Apply Winsorizing
            self.data[col] = np.where(self.data[col] < lower_bound, lower_bound, self.data[col])
            self.data[col] = np.where(self.data[col] > upper_bound, upper_bound, self.data[col])

        print("Winsorized data shape:", self.data.shape)
        print("Mean after outlier treatment:\n",self.num_cols.mean())
        winsorize_path = os.path.join(desired_output_folder)
        os.makedirs(winsorize_path, exist_ok=True)
        output_filename = f'Winsorized_Data_{self.timestamp}.csv'
        output_path = os.path.join(winsorize_path,output_filename)
        self.data.to_csv(output_path, index=False)
        # Box plot after winsorization
        plt.figure(figsize=(12, 8))
        sns.boxplot(data=self.data[self.num_cols.columns])
        plt.title("Box Plot after Winsorization")
        plt.xticks(rotation=45)
        # Adjust layout to prevent x-axis label overlapping
        plt.tight_layout()
        output_path = os.path.join(desired_output_folder,boxplot_folder,f'_BoxPlot_after_Winsorization_{self.timestamp}.png')
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path)
        plt.close()
        plt.close()
        plt.show()
        print("******") 
        return self.data"""
    
    def winsorization(self,y, desired_output_folder):
        # Box plot before winsorization
        plt.figure(figsize=(12, 8))
        num_cols_count=len([col for col in self.num_cols.columns if col != y])
        
        # Create subplots
        fig, axes = plt.subplots(nrows=num_cols_count, ncols=1, figsize=(12, 4*num_cols_count))
        
        for i, col in enumerate(self.num_cols.columns):
            if col != y:
                sns.boxplot(x=self.data[col], ax=axes[i],color='#008080')
                axes[i].set_title(f'Box Plot before Winsorization - {col}')
                axes[i].set_xticks([])  # Remove x-axis ticks for better visibility
                axes[i].set_xlabel('')
            
        plt.tight_layout()
        boxplot_folder = f'Box_Plots_{self.timestamp}'
        output_path = os.path.join(desired_output_folder, boxplot_folder, f'_BoxPlot_before_Winsorization_{self.timestamp}.png')
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path)
        plt.close()

        print("******") 

        for col in self.num_cols.columns:
            if col != y:
                # Calculate the first and third quartiles
                Q1 = self.num_cols[col].quantile(0.25)
                Q3 = self.num_cols[col].quantile(0.75)
                # Set boundaries
                lower_bound = Q1 - 1.5 * (Q3 - Q1)
                upper_bound = Q3 + 1.5 * (Q3 - Q1)
                # Apply Winsorizing
                self.data[col] = np.where(self.data[col] < lower_bound, lower_bound, self.data[col])
                self.data[col] = np.where(self.data[col] > upper_bound, upper_bound, self.data[col])

        print("Winsorized data shape:", self.data.shape)
        print("Mean after outlier treatment:\n", self.num_cols.mean())
        winsorize_path = os.path.join(desired_output_folder)
        os.makedirs(winsorize_path, exist_ok=True)
        output_filename = f'Winsorized_Data_{self.timestamp}.csv'
        output_path = os.path.join(winsorize_path, output_filename)
        self.data.to_csv(output_path, index=False)

        # Box plot after winsorization
        plt.figure(figsize=(12, 8))
        
        # Create subplots
        fig, axes = plt.subplots(nrows=num_cols_count, ncols=1, figsize=(12, 4*num_cols_count))
        
        for i, col in enumerate(self.num_cols.columns):
            if col != y:
                sns.boxplot(x=self.data[col], ax=axes[i],color='#008080')
                axes[i].set_title(f'Box Plot after Winsorization - {col}')
                axes[i].set_xticks([])  # Remove x-axis ticks for better visibility
                axes[i].set_xlabel('')
            
        plt.tight_layout()
        output_path = os.path.join(desired_output_folder, boxplot_folder, f'_BoxPlot_after_Winsorization_{self.timestamp}.png')
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path)
        plt.close()
        print("******") 
        return self.data

    """def perform_encoding(self,desired_output_folder):
        df_encoded = self.data.copy()
        label_mapping={}
        for col in self.data.columns:
            if col not in self.num_cols:
                label_encoder = LabelEncoder()
                df_encoded[col] = label_encoder.fit_transform(self.data[col])
                label_mapping[col] = dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))
        enmap_filename = self.timestamp + "_encode_mapping.pkl"
        encode_mapping_file_path = os.path.join(desired_output_folder, f"{enmap_filename}")
        with open(encode_mapping_file_path, 'wb') as file:
            pickle.dump(label_mapping, file)
        self.data = df_encoded
        return self.data"""

    def perform_encoding(self,y, desired_output_folder):
        df_encoded = self.data.copy()
        label_mapping = {}
        # Step 1: Group and encode quality if present
        if y in df_encoded.columns:
            # Step 1a: Map numeric quality to labels
            quality_mapping = {
                3: "Low", 4: "Low",
                5: "Medium", 6: "Medium", 7: "Medium",
                8: "High", 9: "High"
            }
            df_encoded[y] = df_encoded[y].map(quality_mapping)
            # Step 1b: Map labels to ordinal values
            mapping_quality = {"Low": 0, "Medium": 1, "High": 2}
            df_encoded[y] = df_encoded[y].map(mapping_quality)
        # Step 2: Encode all other categorical columns
        for col in df_encoded.columns:
            if col not in self.num_cols and col != y:
                label_encoder = LabelEncoder()
                df_encoded[col] = label_encoder.fit_transform(df_encoded[col])
                label_mapping[col] = dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))
        # Step 3: Save label encoding mappings
        enmap_filename = self.timestamp + "_encode_mapping.pkl"
        encode_mapping_file_path = os.path.join(desired_output_folder, enmap_filename)
        with open(encode_mapping_file_path, 'wb') as file:
            pickle.dump(label_mapping, file)
        self.data = df_encoded
        return self.data


    
    def split_data(self,desired_output_folder,y,test_size=0.2, random_state=0):
        self.X = self.data.drop([y], axis=1) 
        self.y=self.data[y]
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=test_size, random_state=random_state)
        # Save the data and split variables into a pickle file
        X_train=self.X_train
        X_test=self.X_test
        y_train=self.y_train
        y_test=self.y_test
        datasplit_filename = self.timestamp + "_data_split.pkl"
        data_split_file_path = os.path.join(desired_output_folder, f"{datasplit_filename}")
        with open(data_split_file_path, "wb") as file:
            pickle.dump((X_train,X_test, y_train, y_test), file)
            # Concatenate X_train and y_train into a single DataFrame
        train_data = pd.concat([self.X_train,self.y_train], axis=1)

        # Save training data (X_train and y_train) as a single CSV file
        train_data_csv = os.path.join(desired_output_folder, f"{self.timestamp}_train_data.csv")
        train_data.to_csv(train_data_csv, index=False)

        # Concatenate X_train and y_train into a single DataFrame
        test_data = pd.concat([self.X_test,self.y_test], axis=1)
        # Save training data (X_train and y_train) as a single CSV file
        test_data_csv = os.path.join(desired_output_folder, f"{self.timestamp}_test_data.csv")
        test_data.to_csv(test_data_csv, index=False)
        return self.X_train,self.X_test,self.y_train,self.y_test




"""if __name__ == "__main__":

    pre_processing = preprocessing(Adr_file_path)
    imputeencode=impute_encode(pre_processing)
    timestampfunc=imputeencode.timestamp_func()
    imputed_data=imputeencode.perform_imputation()
    encoded_data=imputeencode.perform_encoding()
    split_data=imputeencode.split_data()"""





