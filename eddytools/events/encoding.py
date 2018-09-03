from collections import namedtuple


Candidate = namedtuple('Candidate', ('timestamp_attribute_id',
                                     'activity_identifier_attribute_id',
                                     'relationship_id'))


class Encoder:

    def __init__(self, engine, meta):
        self.class_id_to_name = dict()
        self.class_name_to_id = dict()
        self.attribute_id_to_name = dict()
        self.attribute_name_to_id = dict()
        self.relationship_id_to_name = dict()
        self.relationship_name_to_id = dict()
        self.load_mappings(engine, meta)

    def load_mappings(self, engine, meta):
        from sqlalchemy import select

        t_class = meta.tables.get('class')
        t_attr = meta.tables.get('attribute_name')
        t_rels = meta.tables.get('relationship')

        self.class_id_to_name = dict()
        self.class_name_to_id = dict()
        self.attribute_id_to_name = dict()
        self.attribute_name_to_id = dict()
        self.relationship_id_to_name = dict()
        self.relationship_name_to_id = dict()

        q = select([t_class.c.id, t_class.c.name])
        res = engine.execute(q)
        for row in res:
            self.class_id_to_name.update({row['id']: row['name']})
            self.class_name_to_id.update({row['name']: row['id']})

        q = (select([t_class.c.id.label('class_id'), t_class.c.name.label('class_name'),
                    t_attr.c.id.label('attribute_id'), t_attr.c.name.label('attribute_name')])
             .select_from(t_class.join(t_attr, t_class.c.id == t_attr.c.class_id)))
        res = engine.execute(q)
        for row in res:
            self.attribute_id_to_name.update({row['attribute_id']: (row['class_name'], row['attribute_name'])})
            self.attribute_name_to_id.update({(row['class_name'], row['attribute_name']): row['attribute_id']})

        q = select([t_rels.c.id, t_rels.c.name])
        res = engine.execute(q)
        for row in res:
            self.relationship_id_to_name.update({row['id']: row['name']})
            self.relationship_name_to_id.update({row['name']: row['id']})

    def transform(self, decoded):
        encoded = []
        for d in decoded:
            timestamp_attribute_id = self.attribute_name_to_id[tuple(d['timestamp_attribute'])]
            activity_identifier_attribute_id = (self.attribute_name_to_id[tuple(d['activity_identifier_attribute'])]
                                                if d['activity_identifier_attribute'] else None)
            relationship_id = (self.relationship_name_to_id[d['relationship']] if d['relationship'] else None)
            encoded.append(Candidate(timestamp_attribute_id=timestamp_attribute_id,
                      activity_identifier_attribute_id=activity_identifier_attribute_id,
                      relationship_id=relationship_id))
        return encoded

    def inverse_transform(self, encoded):
        decoded = []
        for e in encoded:
            d = dict()
            d.update({'timestamp_attribute': self.attribute_id_to_name[e.timestamp_attribute_id]})
            d.update({'activity_identifier_attribute': self.attribute_id_to_name[e.activity_identifier_attribute_id]
                      if e.activity_identifier_attribute_id else None})
            d.update({'relationship': self.relationship_id_to_name[e.relationship_id] if e.relationship_id else None})
            decoded.append(d)
        return decoded
