import pandas as pd
import numpy as np
from sklearn.preprocessing import OrdinalEncoder, LabelEncoder
import mlflow
from sklearn.metrics import classification_report, confusion_matrix, f1_score, precision_score, recall_score,accuracy_score
import mlflow
from mlflow.models import infer_signature
import lightgbm as lgb
from sklearn.model_selection import train_test_split

def read_data(path):
    try:
        df = pd.read_csv(path)
        return df
    except Exception as e:
        print("An error occurred while reading the data:", e)
        return None

def preprocessing(Churn_train_df, Churn_validation_df):
    overweight_threshold = 25
    Churn_train_df['BMI'] = Churn_train_df['Weight'] / (Churn_train_df['Height'] ** 2)
    Churn_train_df['Overweight'] = (Churn_train_df['BMI'] > overweight_threshold).astype(int)
    Churn_train_df = Churn_train_df.drop(columns='BMI')

    Churn_validation_df['BMI'] = Churn_validation_df['Weight'] / (Churn_validation_df['Height'] ** 2)
    Churn_validation_df['Overweight'] = (Churn_validation_df['BMI'] > overweight_threshold).astype(int)
    Churn_validation_df = Churn_validation_df.drop(columns='BMI')

    columns_to_ordinal_encode=Churn_train_df.select_dtypes(exclude=[np.number]).columns
    columns_to_ordinal_encode=columns_to_ordinal_encode.drop('NObeyesdad')

    # Apply Ordinal Encoder to specified columns
    ordinal_encoder = OrdinalEncoder(handle_unknown='use_encoded_value',unknown_value=-1)
    Churn_train_df[columns_to_ordinal_encode] = ordinal_encoder.fit_transform(Churn_train_df[columns_to_ordinal_encode])

    # Apply Label Encoder to target column
    label_encoder = LabelEncoder()
    target_column = 'NObeyesdad'
    Churn_train_df[target_column] = label_encoder.fit_transform(Churn_train_df[target_column])

    Churn_validation_df[columns_to_ordinal_encode] = ordinal_encoder.transform(Churn_validation_df[columns_to_ordinal_encode])

    X = Churn_train_df.drop(columns=[target_column])
    y = Churn_train_df[target_column]

    X_val = Churn_validation_df
    
    return X,y,X_val,label_encoder
       
    
def my_model(X,y,params):
    X_train, X_test, y_train, y_test = train_test_split(X.drop(columns='id'), y, test_size=0.25, random_state=42)

    # Creating XGBClassifier instance
    lgbClassifier_Model = lgb.LGBMClassifier(**params)

    lgbClassifier_Model.fit(X_train, y_train)
    
    # Predict on the test set
    y_pred = lgbClassifier_Model.predict(X_test)

    # Calculate precision, recall, and F1-score
    precision = precision_score(y_test, y_pred,average='micro')
    recall = recall_score(y_test, y_pred,average='micro')
    f1 = f1_score(y_test, y_pred,average='micro')
    accuracy = accuracy_score(y_test, y_pred)
    classification = classification_report(y_test, y_pred)
    confusion = confusion_matrix(y_test, y_pred)

    print(f"Precision: {precision}")
    print(f"Recall: {recall}")
    print(f"F1 Score: {f1}")
    print(f"Accuracy: {accuracy}")
    print(f"Classification Report:\n {classification}")
    print(f"Confusion Matrix:\n {confusion}")
    
    return lgbClassifier_Model,precision,recall,f1,accuracy
    
    
    
def predict(lgbClassifier_Model,X_val,label_encoder):
    from mlflow.models import infer_signature
    y_val_pred = lgbClassifier_Model.predict(X_val.drop(columns='id'))

    y_pred_original = pd.DataFrame(label_encoder.inverse_transform(y_val_pred),columns=['NObeyesdad'])

    submission_df=pd.concat([X_val['id'],y_pred_original],axis=1)

    submission_df.to_csv('submission.csv',index=False)


def save_eval(lgbClassifier_Model,X_train,infer_signature,precision,recall,f1,accuracy):
    mlflow.set_tracking_uri(uri="http://<host>:<port>")
    # Set our tracking server uri for logging
    mlflow.set_tracking_uri(uri="http://127.0.0.1:8080")

    # Create a new MLflow Experiment
    mlflow.set_experiment("MLflow Quickstart")

    # Start an MLflow run
    with mlflow.start_run():
        # Log the hyperparameters
        mlflow.log_params(params)

        # Log the loss metric
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("f1", f1)
        mlflow.log_metric("accuracy", accuracy)

        # Set a tag that we can use to remind ourselves what this run was for
        mlflow.set_tag("Training Info", "Basic LR model for iris data")

        # Infer the model signature
        signature = infer_signature(X_train, lgbClassifier_Model.predict(X_train))

        # Log the model
        model_info = mlflow.sklearn.log_model(
            sk_model=lgbClassifier_Model,
            artifact_path="iris_model",
            signature=signature,
            input_example=X_train,
            registered_model_name="tracking-quickstart",
        )


if '__main__'==__name__:
    import sys
    
    n = int(sys.argv[1])
    print(n+1)
    
    params = {
        'objective': 'multiclass',
        'num_leaves': 10,
        'learning_rate': 0.01,
        'max_depth': -1,
        'min_child_samples': 20,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'reg_alpha': 0.1,
        'reg_lambda': 0.1,
        'n_estimators': 1000,
        'random_state': 42
    }
    
    params['num_leaves']=int(sys.argv[1])
    params['learning_rate']=float(sys.argv[2])
    params['max_depth']=int(sys.argv[3])
    params['min_child_samples']=int(sys.argv[4])
    params['n_estimators']=int(sys.argv[5])
    
    Churn_train_path='/home/aizaz/Ubuntu-Workspace/MLOPS/Data/Raw-Data/train.csv'
    Churn_validation_path = '/home/aizaz/Ubuntu-Workspace/MLOPS/Data/Raw-Data/test.csv'

    Churn_train_df=read_data(Churn_train_path)
    Churn_validation_df=read_data(Churn_validation_path)
    
    X,y,X_val,label_encoder=preprocessing(Churn_train_df, Churn_validation_df)
    
    lgbClassifier_Model,precision,recall,f1,accuracy=my_model(X,y,params)
    
    predict(lgbClassifier_Model,X_val,label_encoder)

    save_eval(lgbClassifier_Model,X.drop(columns='id'),infer_signature,precision,recall,f1,accuracy)