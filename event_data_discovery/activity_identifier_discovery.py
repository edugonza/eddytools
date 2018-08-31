import json
from datetime import datetime
from sqlalchemy import select, and_, or_
from sklearn.model_selection import cross_validate, GridSearchCV, StratifiedKFold
import numpy as np

from .encoding import Encoder, Candidate


class ActivityIdentifierDiscoverer:

    def __init__(self, engine, meta):
        self.engine = engine
        self.meta = meta

        self.encoder = Encoder(self.engine, self.meta)

        self.candidates = None
        self.feature_values = None
        self.y_true = None
        # self.predictions = None

        self.eval_results = None

    def generate_candidates(self, timestamp_attrs):
        """generates and returns the candidate activity identifiers for each event timestamp.
        Only relationships with one degree of separation are considered."""
        self.candidates = []

        for ta in timestamp_attrs:
            self.candidates.append(Candidate(timestamp_attribute_id=ta,
                                             activity_identifier_attribute_id=None,
                                             relationship_id=None))

        engine = self.engine
        meta = self.meta
        t_attr_1 = meta.tables.get('attribute_name').alias()
        t_attr_2 = meta.tables.get('attribute_name').alias()
        t_rels = meta.tables.get('relationship')
        data_types = ('string', 'integer')
        q = (select([t_attr_1.c.id.label('ts_attr'), t_attr_2.c.id.label('aid_attr')])
             .select_from(t_attr_1
                          .join(t_attr_2, t_attr_1.c.class_id == t_attr_2.c.class_id
        )).where(and_(t_attr_1.c.id.in_(timestamp_attrs), t_attr_2.c.type.in_(data_types))))
        result = engine.execute(q)

        for row in result:
            self.candidates.append(Candidate(timestamp_attribute_id=row['ts_attr'],
                                             activity_identifier_attribute_id=row['aid_attr'],
                                             relationship_id=None))

        q = (select([t_attr_1.c.id.label('ts_attr'), t_rels.c.id.label('rel_id'),t_attr_2.c.id.label('aid_attr')])
             .select_from(t_attr_1
                          .join(t_rels, t_attr_1.c.class_id == t_rels.c.source)
                          .join(t_attr_2, t_rels.c.target == t_attr_2.c.class_id))
             .where(and_(t_attr_1.c.id.in_(timestamp_attrs), t_attr_2.c.type.in_(data_types))))
        result = engine.execute(q)
        for row in result:
            self.candidates.append(Candidate(timestamp_attribute_id=row['ts_attr'],
                                             activity_identifier_attribute_id=row['aid_attr'],
                                             relationship_id=row['rel_id']))

    def load_candidates(self, filepath):
        with open(filepath, 'r') as f:
            candidates = json.load(f)
        self.candidates = [Candidate(*c) for c in candidates]

    def compute_features(self, features, filter=True, verbose=0):
        if self.feature_values:
            feature_values = self.feature_values
        else:
            feature_values = [dict() for c in self.candidates]
        for f in features:
            if verbose:
                print(str(datetime.now()) + " computing feature '" + f.__name__ + "'")
            f(self.candidates, feature_values, self.engine, self.meta)
        self.feature_values = feature_values
        if filter:
            self.filter_features(features)

    def load_features(self, filepath, features=None):
        with open(filepath, 'r') as f:
            self.feature_values = json.load(f)
        if features:
            self.filter_features(features)

    def filter_features(self, features):
        feature_names = [f.__name__ for f in features]
        feature_values = self.feature_values
        feature_values_filtered = []
        for fv in feature_values:
            feature_values_filtered.append({name: fv[name] for name in feature_names})
        self.feature_values = feature_values_filtered

    def load_y_true(self, y_true_path):
        with open(y_true_path, "r") as f:
            ground_truth_decoded = json.load(f)
        ground_truth = self.encoder.transform(ground_truth_decoded)
        y_true = [int(c in ground_truth) for c in self.candidates]
        y_true = np.asarray(y_true)
        self.y_true = y_true

    def evaluate(self, names, classifiers, n_splits=5, verbose=0):

        eval_results = list()

        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=1)
        scoring = ['precision', 'recall', 'f1']

        for name, clf in zip(names, classifiers):
            print(str(datetime.now()) + ': evaluating ' + name)
            eval_results.append(cross_validate(clf, self.feature_values, self.y_true, scoring=scoring, cv=cv,
                                                verbose=verbose))

        for name, res in zip(names, eval_results):
            print('evaluation results for predictor ' + name + ':')
            print('average precision: ' + str(res['test_precision'].mean()))
            print('average recall: ' + str(res['test_recall'].mean()))
            print('average f1: ' + str(res['test_f1'].mean()))
            print()

        return eval_results

    def tune_params(self, classifiers, parameters, scoring = 'f1', n_splits=5, verbose=0):

        tuning_results = list()

        for clf, params in zip(classifiers, parameters):
            tuner = GridSearchCV(clf, params, scoring=scoring, cv=n_splits, refit=False, verbose=verbose,
                                 return_train_score=False)
            tuner.fit(self.feature_values, self.y_true)
            tuning_results.append({
                'cv_results': tuner.cv_results_,
                'best_score': tuner.best_score_,
                'best_params': tuner.best_params_,
                'best_index': tuner.best_index_,
            })

        return tuning_results
