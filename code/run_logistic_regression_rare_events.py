import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from user_defined_functions import make_frequency_plot,

# ~ Read in our data ~ #
# path_to_data='/home/ryan/Downloads/creditcard.csv'
path_to_data='C:/Users/ryanl/Downloads/creditcardfraud/creditcard.csv'
df=pd.read_csv(path_to_data)

# ~ Split data into dependent and indepent variables ~ #
df=df[[x for x in df.columns if x!='Time']]
iv=df['Class']
dv=df[[x for x in df.columns if x!='Class']]

# ~~ Make frequency plot ~~ #
make_frequency_plot(iv)

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
#                  Setup Model Pipeline                     #
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #

# ~~~~ Pipeline + Data Transform Libraries ~~~~ #
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression

# (1) : Seperate numeric and categorical variables
# (2) : Create transforms (imputation + standardization) for both types
numeric_features = list(dv.dtypes[((dv.dtypes=='int64') | (dv.dtypes=='float64')) ].index.values)
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())])

categorical_features = list(dv.dtypes[((dv.dtypes=='object')) ].index.values)
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))])

# (3) : Concatenate columns together and apply transformations
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)])

# (4) : Create pipeline with preprocessing + estimation steps
clf = Pipeline(steps=[('preprocessor', preprocessor),
                      ('classifier', LogisticRegression(solver='lbfgs',max_iter=1000))])

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
# ~ Run logisitic regression using Weighted MLE v. Stratified ~ #
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #

# ~~ Model estimation + selection libraries ~~ #
from sklearn.model_selection import cross_validate, KFold
from sklearn.metrics import accuracy_score, recall_score, average_precision_score, f1_score, make_scorer

# Set up 10 fold cross-validation with specified random seed
seed=1
n_splits=10

# Initialize KFold Object
kfold_obj = KFold(n_splits=n_splits, random_state=seed)

# Lists to hold scores for model runs
accuracy_list=[]
recall_list=[]
average_precision_list=[]
f1_weighted_list=[]

# Loop through weights
weights=np.arange(0.00,1.00,0.05)
for sample_weight in weights:

    # ~ Model with no weighting ~ #
    if sample_weight==0:
        scores = cross_validate(clf,dv,iv,cv=kfold_obj,
                        scoring={'accuracy':make_scorer(accuracy_score),
                                 'recall':make_scorer(recall_score),
                                 'average_precision_score':make_scorer(average_precision_score)})
        # ~ Calculate different scoring rules ~ #
        mean_accuracy_weighted=np.mean(scores['test_accuracy'])
        mean_recall_weighted=np.mean(scores['test_recall'])
        mean_avg_precision_weighted=np.mean(scores['test_average_precision_score'])
        mean_accuracy_oversample=mean_accuracy_weighted
        mean_recall_oversample=mean_recall_weighted
        mean_avg_precision_oversample=mean_avg_precision_weighted



    # ~ Model with weighting ~ #
    else:
        # ~ Weighted Likelihood ~ #
        weighted_scores = cross_validate(clf,dv,iv,cv=kfold_obj,
                        scoring={'accuracy':make_scorer(accuracy_score),
                                 'recall':make_scorer(recall_score),
                                 'average_precision_score':make_scorer(average_precision_score)},
                       fit_params={'classifier__sample_weight':[[1-sample_weight,sample_weight][x==1] for x in iv]})

        # ~ Calculate different scoring rules ~ #
        mean_accuracy_weighted=np.mean(weighted_scores['test_accuracy'])
        mean_recall_weighted=np.mean(weighted_scores['test_recall'])
        mean_avg_precision_weighted=np.mean(weighted_scores['test_average_precision_score'])


        # ~ Stratified Sample ~ #
        oversample_kfold_obj = oversample_KFold(n_splits=n_splits,random_state=seed,sample_weight=sample_weight)
        oversampling_scores = cross_validate(clf,dv,iv,cv=oversample_kfold_obj.yield_sample(df),
                                        scoring={'accuracy':make_scorer(accuracy_score),
                                                 'recall':make_scorer(recall_score),
                                                 'average_precision_score':make_scorer(average_precision_score)})

        # ~ Calculate different scoring rules ~ #
        mean_accuracy_oversample=np.mean(oversampling_scores['test_accuracy'])
        mean_recall_oversample=np.mean(oversampling_scores['test_recall'])
        mean_avg_precision_oversample=np.mean(oversampling_scores['test_average_precision_score'])

    # ~~~ Append data to list ~~~ #
    accuracy_list.append((sample_weight,mean_accuracy_weighted,mean_accuracy_oversample))
    recall_list.append((sample_weight,mean_recall_weighted,mean_recall_oversample))
    average_precision_list.append((sample_weight,mean_avg_precision_weighted,mean_avg_precision_oversample))
    
# ~~~ Form data frame of scores ~~~ #
scoresDF=pd.DataFrame([(x[0],x[1],'precision','wmle') for x in average_precision_list],
                      columns=['weight','score_value','score','estimator'])
scoresDF=scoresDF.append(pd.DataFrame([(x[0],x[2],'precision','stratified') for x in average_precision_list],
                         columns=['weight','score_value','score','estimator']))
scoresDF=scoresDF.append(pd.DataFrame([(x[0],x[1],'recall','wmle') for x in recall_list],
                      columns=['weight','score_value','score','estimator']))
scoresDF=scoresDF.append(pd.DataFrame([(x[0],x[2],'recall','stratified') for x in recall_list],
                         columns=['weight','score_value','score','estimator']))
scoresDF=scoresDF.append(pd.DataFrame([(x[0],x[1],'accuracy','wmle') for x in accuracy_list],
                      columns=['weight','score_value','score','estimator']))
scoresDF=scoresDF.append(pd.DataFrame([(x[0],x[2],'accuracy','stratified') for x in accuracy_list],
                         columns=['weight','score_value','score','estimator']))


# ~ Scores dataframe ~ #
scoresDF.head()


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
# Plot model selection scores
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #

# ~ Accuracy ~ #
make_scores_plot(scoresDF,'accuracy')

# ~ Recall ~ #
make_scores_plot(scoresDF,'recall')

# ~ Precision ~ #
make_scores_plot(scoresDF,'precision')