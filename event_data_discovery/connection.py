from sqlalchemy import create_engine, MetaData


def connect_to_openslex(openslex_file_path):

    engine = create_engine('sqlite:///{p}'.format(p=openslex_file_path))
    meta = MetaData()
    meta.reflect(bind=engine)

    return engine, meta
