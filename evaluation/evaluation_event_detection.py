from datetime import datetime

from eddytools.events.activity_identifier_discovery import ActivityIdentifierDiscoverer
from eddytools.events.activity_identifier_predictors import make_sklearn_pipeline, OnePerTS

# ____________________________________________________________________________________________________________________

from sklearn.utils import compute_sample_weight, compute_class_weight

from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import make_scorer, fbeta_score

# from sknn.mlp import Classifier as NNClassifier
# from sknn.mlp import Layer

from scipy.stats import randint as sp_randint, expon as sp_expon, uniform as sp_uniform
from xgboost.sklearn import XGBClassifier, XGBRegressor

# ____________________________________________________________________________________________________________________

mimic_files = {
    'openslex': 'output/mimic/sample-mimic.slexmm',
    'timestamps': 'output/mimic/evdisc/event_timestamps.json',
    'candidates': 'output/mimic/evdisc/candidates_in_table.json',
    'feature_values': 'output/mimic/evdisc/features_in_table.json',
    # 'candidates': 'mimic/candidates_lookup.json',
    # 'feature_values': 'mimic/features_lookup.json',
    'y_true': 'data/mimic/ground_truth.json',
}

# ____________________________________________________________________________________________________________________

adw_files = {
    'openslex': 'data/adw/extracted-adw-mm.slexmm',
    'timestamps': 'output/adw/evdisc/timestamps.json',
    'candidates': 'output/adw/evdisc/candidates_in_table.json',
    'feature_values': 'output/adw/evdisc/features_in_table.json',
    # 'candidates': 'adw/candidates_lookup.json',
    # 'feature_values': 'adw/features_lookup.json',
    'y_true': 'data/adw/ground_truth.json',
}

# aux_files = adw_files
# adw_files = mimic_files
# mimic_files = aux_files

# ____________________________________________________________________________________________________________________

aid_mimic = ActivityIdentifierDiscoverer(mimic_files['openslex'])
# mimic_timestamp_atts = aid_mimic.get_timestamp_attributes()
# aid_mimic.save_timestamp_attributes(mimic_timestamp_atts, mimic_files['timestamps'])
# aid_mimic.generate_candidates(mimic_files['timestamps'], ['in_table'])
# aid_mimic.generate_candidates(mimic_files['timestamps'], ['lookup'])
# aid_mimic.save_candidates(mimic_files['candidates'])
aid_mimic.load_candidates(mimic_files['candidates'])
# aid_mimic.compute_features(features='filtered')
# aid_mimic.save_features(mimic_files['feature_values'])
aid_mimic.load_features(mimic_files['feature_values'])
aid_mimic.load_y_true(mimic_files['y_true'])

#
aid_adw = ActivityIdentifierDiscoverer(adw_files['openslex'])
# adw_timestamp_atts = aid_adw.get_timestamp_attributes()
# aid_adw.save_timestamp_attributes(adw_timestamp_atts, adw_files['timestamps'])
# aid_adw.generate_candidates(adw_files['timestamps'], ['in_table'])
# aid_adw.generate_candidates(adw_files['timestamps'], ['lookup'])
# aid_adw.save_candidates(adw_files['candidates'])
aid_adw.load_candidates(adw_files['candidates'])
# aid_adw.compute_features(features='filtered')
# aid_adw.save_features(adw_files['feature_values'])
aid_adw.load_features(adw_files['feature_values'])
aid_adw.load_y_true(adw_files['y_true'])

eval_set = [(DictVectorizer(sparse=False, sort=True).fit_transform(aid_adw.feature_values), aid_adw.y_true)]

# ____________________________________________________________________________________________________________________

sample_weights = compute_sample_weight('balanced', aid_mimic.y_true)
pos_weight = compute_class_weight('balanced', [0, 1], aid_mimic.y_true)

tuning_params = [
    # {
    #     'name': "Nearest Neighbors",
    #     'predictor': make_sklearn_pipeline(KNeighborsClassifier()),
    #     'parameters': {
    #         'clf__n_neighbors': sp_randint(2, 20),
    #         'clf__weights': ['uniform', 'distance'],
    #     },
    #     'n_iter': 1000,
    #     'fit_params': None,
    # },
    # {
    #     'name': "Linear SVM",
    #     'predictor': make_sklearn_pipeline(SVC(kernel="linear", class_weight='balanced', random_state=1)),
    #     'parameters': {
    #         'clf__C': sp_expon(),
    #     },
    #     'n_iter': 1000,
    #     'fit_params': None,
    # },
    # {
    #     'name': "RBF SVM",
    #     'predictor': make_sklearn_pipeline(SVC(class_weight='balanced', random_state=1)),
    #     'parameters': {
    #         'clf__C': sp_expon(),
    #     },
    #     'n_iter': 1000,
    #     'fit_params': None,
    # },
    # {
    #     'name': "Decision Tree",
    #     'predictor': make_sklearn_pipeline(DecisionTreeClassifier(class_weight='balanced', random_state=1)),
    #     'parameters': {
    #         'clf__min_weight_fraction_leaf': sp_uniform(loc=0, scale=0.5),
    #     },
    #     'n_iter': 1000,
    #     'fit_params': None,
    # },
    # {
    #     'name': "Random Forest",
    #     'predictor': make_sklearn_pipeline(RandomForestClassifier(class_weight='balanced', random_state=1)),
    #     'parameters': {
    #         'clf__n_estimators': sp_randint(10, 500),
    #         'clf__max_features': sp_uniform(),
    #         'clf__min_weight_fraction_leaf': sp_uniform(loc=0, scale=0.5),
    #     },
    #     'n_iter': 10,
    #     'fit_params': None,
    # },
    # {
    #     'name': "AdaBoost",
    #     'predictor': make_sklearn_pipeline(
    #         AdaBoostClassifier(DecisionTreeClassifier(class_weight='balanced', random_state=1), random_state=1)),
    #     'parameters': {
    #         'clf__n_estimators': sp_randint(10, 500),
    #         'clf__base_estimator__min_weight_fraction_leaf': sp_uniform(loc=0, scale=0.5),
    #     },
    #     'n_iter': 10,
    #     'fit_params': None,
    # },
    # {
    #     'name': 'Neural Network',
    #     'predictor': make_sklearn_pipeline(
    #         NNClassifier(
    #             layers=[
    #                 Layer("Rectifier", units=25),
    #                 Layer("Softmax")
    #             ],
    #             n_iter=25,
    #             random_state=1
    #         )
    #     ),
    #     'parameters': {
    #         'clf__hidden0__units': sp_randint(2, 50),
    #         'clf__learning_rate': sp_expon(scale=.001)
    #     },
    #     'n_iter': 100,
    #     'fit_params': {
    #         'clf__w': sample_weights
    #     },
    # },
    {
        'name': "XGBoost",
        'predictor': make_sklearn_pipeline(
            XGBClassifier(random_state=1, max_depth=6, n_estimators=100, scale_pos_weight=pos_weight[1])),  # , subsample=0.5)),
        'parameters': {
            # 'clf__n_estimators': sp_randint(1, 500),
            # 'clf__base_estimator__min_weight_fraction_leaf': sp_uniform(loc=0, scale=0.5),
            # 'clf__max_depth': sp_randint(1, 10),
            # 'clf__subsample': sp_uniform(),
        },
        'n_iter': 1,
        'fit_params': {
            # 'clf__sample_weight': sample_weights
            # 'clf__eval_metric': "rmse",
            # 'clf__eval_set': eval_set,
        },
    },
    # {
    #     'name': "Random",
    #     'predictor': make_sklearn_pipeline(
    #         OnePerTS(random_state=1)),
    #     'parameters': {
    #     },
    #     'n_iter': 1,
    #     'fit_params': None,
    # },
]
scoring = {
    'precision': 'precision',
    'recall': 'recall',
    'f1': 'f1',
    'f.5': make_scorer(fbeta_score, beta=2),
    'f2': make_scorer(fbeta_score, beta=2),
}
n_splits = 5
refit = 'neg_mean_squared_error'
verbose = 1

t0 = datetime.now()
print(str(t0) + ': starting parameter tuning')
aid_mimic.tune_params(tuning_params, scoring, n_splits=n_splits, refit=refit,
                      verbose=verbose)
t1 = datetime.now()
print(str(t1) + ': parameter tuning finished')
print('time elapsed: ' + str(t1 - t0))

# ____________________________________________________________________________________________________________________

aid_mimic.save_tuning_results('tuning_results_all.pkl')
aid_mimic.save_best_predictors('best_predictors_all.pkl')
# aid_mimic.score()
# aid_mimic.save_scores('scores_mimic.pkl')

# ____________________________________________________________________________________________________________________

aid_adw.load_predictors('best_predictors_all.pkl')
# aid_adw.score()
# aid_adw.save_scores('scores_all.pkl')
aid_adw.score_proba()
aid_adw.save_scores('scores_all_proba.pkl')
import pickle
from pprint import pprint
# adw_scores = pickle.load(open('scores_all.pkl', mode='rb'))
# pprint(adw_scores)
adw_scores_proba = pickle.load(open('scores_all_proba.pkl', mode='rb'))
pprint(adw_scores_proba)
