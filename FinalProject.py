#import libraries
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, OrdinalEncoder, LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import Pipeline
import seaborn as sns
import matplotlib.pyplot as plt
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler, SMOTE
from xgboost import XGBClassifier
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import AdaBoostClassifier,RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
import joblib
from sklearn.preprocessing import FunctionTransformer
# TAKE THE DATA FROM EXCEL
data = pd.read_csv('C:\\Users\\artingale\\Desktop\\ada proje\\pages\\bank-additional-full.csv',sep=';')

#Column Transfer Step Of Preprocessing
# IMPORTANT NOTE: We could have include column transfer part in pipeline  and we tried that actually
# but it restrains our ability to control columns and also label_binazier and label encoder gives error becuase of the implementation of scikit-learn
# therefore we did column transfer before the pipeline.

def apply_preprocessing(data):
    # Initialize encoders
    label_encoder = LabelEncoder()
    label_binarizer = LabelBinarizer()

    # Encode 'job' column
    job_order = [['unknown','unemployed','student','retired','housemaid','services','blue-collar','technician','self-employed','management','admin.','entrepreneur']]
    ordinal_encoder_job = OrdinalEncoder(categories=job_order)
    data['job'] = ordinal_encoder_job.fit_transform(data[['job']])

    # Encode 'marital' column
    data['marital'] = label_encoder.fit_transform(data['marital'])

    # Encode 'education' column
    education_order = [['illiterate','unknown','basic.4y','basic.6y','basic.9y','high.school','university.degree','professional.course']]
    ordinal_encoder_education = OrdinalEncoder(categories=education_order)
    data['education'] = ordinal_encoder_education.fit_transform(data[['education']])

    # Encode 'default' column
    default_order = [['no','unknown','yes']]
    ordinal_encoder_default = OrdinalEncoder(categories=default_order)
    data['default'] = ordinal_encoder_default.fit_transform(data[['default']])

    # Encode 'housing' column
    housing_order = [['no','unknown','yes']]
    ordinal_encoder_housing = OrdinalEncoder(categories=housing_order)
    data['housing'] = ordinal_encoder_housing.fit_transform(data[['housing']])

    # Encode 'loan' column
    loan_order = [['no','unknown','yes']]
    ordinal_encoder_loan = OrdinalEncoder(categories=loan_order)
    data['loan'] = ordinal_encoder_loan.fit_transform(data[['loan']])

    # Encode 'contact' column using LabelBinarizer
    data['contact'] = label_binarizer.fit_transform(data['contact'])

    # Encode 'month' column
    data['month'] = label_encoder.fit_transform(data['month'])

    # Encode 'day_of_week' column
    data['day_of_week'] = label_encoder.fit_transform(data['day_of_week'])

    # Encode 'poutcome' column
    poutcome_order = [['nonexistent','failure','success']]
    ordinal_encoder_poutcome = OrdinalEncoder(categories=poutcome_order)
    data['poutcome'] = ordinal_encoder_poutcome.fit_transform(data[['poutcome']])
    data['duration_square']=data['duration'] ** 2
    data=data.drop(columns='nr.employed')

    return data


data = apply_preprocessing(data)
label_binazier = LabelBinarizer()
data['y'] = label_binazier.fit_transform(data['y'])

# Train Test Split Part
X = data.drop(columns='y', axis=1)
Y = data['y']
X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size=0.20,random_state=15)
# Handling Unbalance Data Part of Preprocessing
ros = RandomOverSampler(random_state=18)
X_train, Y_train = ros.fit_resample(X_train, Y_train)

#smote = SMOTE(random_state=42)
#X_train, Y_train = smote.fit_resample(X_train, Y_train)

#rus = RandomUnderSampler(random_state=15)
#X_train, Y_train = rus.fit_resample(X_train, Y_train)

#Creating Pipeline
# IMPORTANT NOTE: We could have include column transfer part in here and we tried that actually
# but it restrains our ability to control columns and also label_binazier and label encoder gives error becuase of the implementation of scikit-learn
# therefore we did column transfer before the pipeline.
pipelinexgb = make_pipeline(
    FunctionTransformer(apply_preprocessing),
    StandardScaler(),
    XGBClassifier()
)
xgb_param_grid = {
    'xgbclassifier__learning_rate': [0.01, 0.1, 0.2],
    'xgbclassifier__n_estimators': [50, 100, 200],
    'xgbclassifier__max_depth': [3, 5, 7],
    'xgbclassifier__objective': ['binary:logistic']
}

# Create and fit the grid search
grid = GridSearchCV(pipelinexgb, xgb_param_grid, cv=2)
grid.fit(X_train, Y_train)

# Get the best pipeline (including the best model with hyperparameters)
best_pipeline = grid.best_estimator_

# Save the entire pipeline (including preprocessing steps and best model) to a file
joblib.dump(best_pipeline, 'xgb_pipeline_filename.joblib')

# Make predictions on the test set using the best pipeline
predictions = best_pipeline.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(Y_test, predictions)
print(f'Accuracy: {accuracy}')

recall = recall_score(Y_test, predictions)
print(f"Recall: {recall}")

precision = precision_score(Y_test, predictions)
print(f"Precision: {precision}")

f1 = f1_score(Y_test, predictions)
print(f"F1 Score: {f1}")


'''Accuracy: 0.896091284292304
recall: 0.7991543340380549
precision: 0.5316455696202531
f1 score : 0.6385135135135135'''



