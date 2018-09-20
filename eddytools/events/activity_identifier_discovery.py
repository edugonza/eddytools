import json
import numpy as np
import pickle
from datetime import datetime

from eddytools.casenotions import get_all_classes
from sqlalchemy.schema import Table, MetaData, Column
from sqlalchemy import select, and_, or_
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import precision_score, recall_score, f1_score, fbeta_score

from .encoding import Encoder, Candidate
from .activity_identifier_feature_functions import filtered as filtered_features
from sqlalchemy import create_engine, MetaData

def connect_to_openslex(openslex_file_path):

    engine = create_engine('sqlite:///{p}'.format(p=openslex_file_path))
    meta = MetaData()
    meta.reflect(bind=engine)

    return engine, meta


class ActivityIdentifierDiscoverer:

    def __init__(self, openslex_filepath):
        self.engine, self.meta = connect_to_openslex(openslex_filepath)

        self.encoder = Encoder(self.engine, self.meta)

        self.candidates = None
        self.feature_values = None
        self.y_true = None

        self.predictors = None
        self.tuning_results = None

    def get_timestamp_attributes(self, classes=None):
        timestamp_attrs = []

        tb_class: Table = self.meta.tables['class'].alias('cl')
        tb_att_n: Table = self.meta.tables['attribute_name'].alias('at')
        c_att_type: Column = tb_att_n.columns['type']
        c_att_name: Column = tb_att_n.columns['name']
        c_att_id: Column = tb_att_n.columns['id']
        c_cl_name: Column = tb_class.columns['name']

        class_names = []
        if not classes:
            cls = get_all_classes(self.engine, self.meta)
            class_names = [c['name'] for c in cls]
        else:
            class_names = classes

        query = select([c_att_id]). \
            select_from(tb_class.join(tb_att_n)). \
            where(and_(c_att_type == 'timestamp',
                       c_cl_name.in_(class_names)))
        ts_fields = self.engine.execute(query)
        for ts in ts_fields:
            timestamp_attrs.append(ts[0])

        return timestamp_attrs

    def save_timestamp_attributes(self, timestamp_attrs, path):
        json.dump(timestamp_attrs, open(path, mode='wt'))

    def generate_candidates(self, timestamp_attrs_filepath, candidate_types):
        """generates and returns the candidate activity identifiers for each event timestamp.
        Only relationships with one degree of separation are considered."""
        self.candidates = []

        with open(timestamp_attrs_filepath, 'r') as f:
            timestamp_attrs = json.load(f)

        engine = self.engine
        meta = self.meta

        t_attr_1 = meta.tables.get('attribute_name').alias()
        t_attr_2 = meta.tables.get('attribute_name').alias()
        t_rels = meta.tables.get('relationship')
        data_types = ['string', 'integer']

        if type(candidate_types) == str:
            candidate_types = [candidate_types]

        if 'in_table' in candidate_types:
            q = (select([t_attr_1.c.id.label('ts_attr'), t_attr_2.c.id.label('aid_attr')])
                 .select_from(t_attr_1
                              .join(t_attr_2, t_attr_1.c.class_id == t_attr_2.c.class_id))
                 .where(and_(t_attr_1.c.id.in_(timestamp_attrs), t_attr_2.c.type.in_(data_types))))
            result = engine.execute(q)

            for row in result:
                self.candidates.append(Candidate(timestamp_attribute_id=row['ts_attr'],
                                                 activity_identifier_attribute_id=row['aid_attr'],
                                                 relationship_id=None))

        if 'lookup' in candidate_types:
            q = (select([t_attr_1.c.id.label('ts_attr'), t_rels.c.id.label('rel_id'), t_attr_2.c.id.label('aid_attr')])
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

    def save_candidates(self, filepath):
        with open(filepath, 'w') as f:
            json.dump(self.candidates, f, indent=4, sort_keys=True)

    def compute_features(self, features, filter_=True, verbose=0):
        if self.feature_values:
            feature_values = self.feature_values
        else:
            feature_values = [dict() for c in self.candidates]
        if features == 'filtered':
            features = filtered_features
        for f in features:
            if verbose:
                print(str(datetime.now()) + " computing feature '" + f.__name__ + "'")
            f(self.candidates, feature_values, self.engine, self.meta)
        self.feature_values = feature_values
        if filter_:
            self.filter_features(features)

    def load_features(self, filepath, features=None):
        with open(filepath, 'rt') as f:
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

    def save_features(self, filepath):
        with open(filepath, 'wt') as f:
            json.dump(self.feature_values, f, indent=4, sort_keys=True)

    def load_y_true(self, y_true_path):
        with open(y_true_path, "rt") as f:
            ground_truth_decoded = json.load(f)
        ground_truth = self.encoder.transform(ground_truth_decoded)
        y_true = [int(c in ground_truth) for c in self.candidates]
        y_true = np.asarray(y_true)
        self.y_true = y_true

    def predictors_from_tuning_results(self):
        best_predictors = [{
            'name': tr['name'],
            'predictor': tr['tuner'].best_estimator_
        } for tr in self.tuning_results]
        self.predictors = best_predictors
        return self.predictors

    def predict_proba(self):
        for pr in self.predictors:
            # pr['predictions'] = pr['predictor'].predict(self.feature_values)
            pr['predictions'] = pr['predictor'].predict_proba(self.feature_values)
        return self.predictors

    def predict(self):
        for pr in self.predictors:
            pr['predictions'] = pr['predictor'].predict(self.feature_values)
        return self.predictors

    def score(self):
        self.predict()
        for pr in self.predictors:
            pr['scores'] = {
                'precision': precision_score(self.y_true, pr['predictions']),
                'recall': recall_score(self.y_true, pr['predictions']),
                'f1': f1_score(self.y_true, pr['predictions']),
                'f.5': fbeta_score(self.y_true, pr['predictions'], beta=.5),
                'f2': fbeta_score(self.y_true, pr['predictions'], beta=2),
            }
        return self.predictors

    def score_proba(self):
        self.predict_proba()
        for pr in self.predictors:
            ts_y = dict()
            for i, (c, p) in enumerate(zip(self.candidates, pr['predictions'])):
                p_pos = p[1]
                if c[0] not in ts_y:
                    ts_y[c[0]] = []
                ts_y[c[0]].append((i, p_pos))

            pred = np.zeros(shape=self.y_true.__len__())
            for ts, probs in ts_y.items():
                sorted_probs = sorted(probs, key=lambda x: x[1], reverse=True)
                for (i, p) in sorted_probs[:1]:
                    if p > 0.5:
                        pred[i] = 1

            pr['scores'] = {
                'precision': precision_score(self.y_true, pred),
                'recall': recall_score(self.y_true, pred),
                'f1': f1_score(self.y_true, pred),
                'f.5': fbeta_score(self.y_true, pred, beta=.5),
                'f2': fbeta_score(self.y_true, pred, beta=2),
            }
        return self.predictors

    def save_predictors(self, filepath):
        predictors = [{'name': pr['name'], 'predictor': pr['predictor']} for pr in self.predictors]
        with open(filepath, 'wb') as f:
            pickle.dump(predictors, f)

    def save_scores(self, filepath):
        scores = [{'name': pr['name'], 'scores': pr['scores']} for pr in self.predictors]
        with open(filepath, 'wb') as f:
            pickle.dump(scores, f)

    def tune_params(self, tuning_params, scoring, n_splits=5, refit=False,
                    verbose=0):

        if not self.tuning_results:
            self.tuning_results = list()
        tuning_results = self.tuning_results
        for tp in tuning_params:
        # for name, clf, params, n_iter, fit_params in zip(names, classifiers, parameters, n_iters, fit_parameters):
            if verbose:
                print(str(datetime.now()) + ': fitting ' + tp['name'])
            tuner = RandomizedSearchCV(tp['predictor'], tp['parameters'], n_iter=tp['n_iter'], scoring=scoring,
                                       cv=n_splits, refit=refit, verbose=verbose, return_train_score=False,
                                       random_state=1)
            if tp.get('fit_params'):
                tuner.fit(self.feature_values, self.y_true, **tp.get('fit_params'))
            else:
                tuner.fit(self.feature_values, self.y_true)
            tuning_results.append({
                'name': tp['name'],
                'predictor': tp['predictor'],
                'parameters': tp['parameters'],
                'n_iter': tp['n_iter'],
                'fit_params': tp.get('fit_params'),
                'tuner': tuner
            })

        return tuning_results

    def save_tuning_results(self, file_path):
        with open(file_path, 'wb') as f:
            pickle.dump(self.tuning_results, f)

    def save_best_predictors(self, file_path):
        best_predictors = self.predictors_from_tuning_results()
        with open(file_path, 'wb') as f:
            pickle.dump(best_predictors, f)

    def load_tuning_results(self, file_path):
        with open(file_path, 'rb') as f:
            self.tuning_results = pickle.load(f)

    def load_predictors(self, file_path):
        with open(file_path, 'rb') as f:
            predictors = pickle.load(f)
        self.predictors = [{'name': pr['name'], 'predictor': pr['predictor']} for pr in predictors]
