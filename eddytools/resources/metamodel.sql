--
-- File generated with SQLiteStudio v3.0.7 on Thu Mar 3 22:37:36 2016
--
-- Text encoding used: UTF-8
--
--PRAGMA foreign_keys = off;
--BEGIN TRANSACTION;

-- Table: event_to_object_version
CREATE TABLE IF NOT EXISTS event_to_object_version (event_id INTEGER REFERENCES event (id) NOT NULL, object_version_id INTEGER REFERENCES object_version (id) NOT NULL, label TEXT, PRIMARY KEY (event_id, object_version_id));

-- Table: case_to_log
CREATE TABLE IF NOT EXISTS case_to_log (case_id INTEGER REFERENCES "case" (id), log_id INTEGER REFERENCES log (id), PRIMARY KEY (case_id, log_id));

-- Table: relation
CREATE TABLE IF NOT EXISTS relation (id INTEGER PRIMARY KEY AUTOINCREMENT, source_object_version_id INTEGER REFERENCES object_version (id), target_object_version_id INTEGER REFERENCES object_version (id), relationship_id INTEGER REFERENCES relationship (id), start_timestamp INTEGER, end_timestamp INTEGER);

-- Table: relationship
CREATE TABLE IF NOT EXISTS relationship (id INTEGER PRIMARY KEY AUTOINCREMENT, name TEXT NOT NULL, source INTEGER REFERENCES class (id), target INTEGER REFERENCES class (id));

-- Table: class
CREATE TABLE IF NOT EXISTS class (id INTEGER PRIMARY KEY AUTOINCREMENT, datamodel_id INTEGER REFERENCES datamodel (id), name TEXT NOT NULL);

-- Table: attribute_name
CREATE TABLE IF NOT EXISTS attribute_name (id INTEGER PRIMARY KEY AUTOINCREMENT, name TEXT, class_id INTEGER REFERENCES class (id), type TEXT);

-- Table: classifier
CREATE TABLE IF NOT EXISTS classifier (id INTEGER PRIMARY KEY AUTOINCREMENT, log_id INTEGER REFERENCES log (id), name TEXT);

-- Table: event_attribute_name
CREATE TABLE IF NOT EXISTS event_attribute_name (id INTEGER PRIMARY KEY AUTOINCREMENT, name TEXT, type TEXT);

-- Table: attribute_value
CREATE TABLE IF NOT EXISTS attribute_value (id INTEGER PRIMARY KEY AUTOINCREMENT, object_version_id INTEGER REFERENCES object_version (id), attribute_name_id INTEGER REFERENCES attribute_name (id), value TEXT, type TEXT);

-- Table: activity_to_process
CREATE TABLE IF NOT EXISTS activity_to_process (process_id INTEGER REFERENCES process (id), activity_id INTEGER REFERENCES activity (id), PRIMARY KEY (process_id, activity_id));

-- Table: event
CREATE TABLE IF NOT EXISTS event (id INTEGER PRIMARY KEY AUTOINCREMENT, activity_instance_id INTEGER REFERENCES activity_instance (id), ordering INTEGER, timestamp INTEGER, lifecycle TEXT, resource TEXT);

-- Table: object_version
CREATE TABLE IF NOT EXISTS object_version (id INTEGER PRIMARY KEY AUTOINCREMENT, object_id INTEGER REFERENCES object (id), start_timestamp INTEGER, end_timestamp INTEGER);

-- Table: activity_instance_to_case
CREATE TABLE IF NOT EXISTS activity_instance_to_case (case_id INTEGER REFERENCES "case" (id) NOT NULL, activity_instance_id INTEGER REFERENCES activity_instance (id) NOT NULL, PRIMARY KEY (case_id, activity_instance_id));

-- Table: activity_instance
CREATE TABLE IF NOT EXISTS activity_instance (id INTEGER PRIMARY KEY AUTOINCREMENT, activity_id INTEGER REFERENCES activity (id));

-- Table: datamodel
CREATE TABLE IF NOT EXISTS datamodel (id INTEGER PRIMARY KEY AUTOINCREMENT, name TEXT);

-- Table: process
CREATE TABLE IF NOT EXISTS process (id INTEGER PRIMARY KEY AUTOINCREMENT, name TEXT);

-- Table: case
CREATE TABLE IF NOT EXISTS "case" (id INTEGER PRIMARY KEY AUTOINCREMENT, name TEXT);

-- Table: case_attribute_name
CREATE TABLE IF NOT EXISTS case_attribute_name (id INTEGER PRIMARY KEY AUTOINCREMENT, name TEXT, type TEXT);

-- Table: log_attribute_value
CREATE TABLE IF NOT EXISTS log_attribute_value (id INTEGER PRIMARY KEY AUTOINCREMENT, log_attribute_name_id INTEGER REFERENCES log_attribute_name (id), log_id INTEGER REFERENCES log (id), value TEXT, type TEXT);

-- Table: log
CREATE TABLE IF NOT EXISTS log (id INTEGER PRIMARY KEY AUTOINCREMENT, process_id INTEGER REFERENCES process (id), name TEXT);

-- Table: log_attribute_name
CREATE TABLE IF NOT EXISTS log_attribute_name (id INTEGER PRIMARY KEY AUTOINCREMENT, name TEXT, type TEXT);

-- Table: case_attribute_value
CREATE TABLE IF NOT EXISTS case_attribute_value (id INTEGER PRIMARY KEY AUTOINCREMENT, case_id INTEGER REFERENCES "case" (id), case_attribute_name_id INTEGER REFERENCES case_attribute_name (id), value TEXT, type TEXT);

-- Table: event_attribute_value
CREATE TABLE IF NOT EXISTS event_attribute_value (id INTEGER PRIMARY KEY AUTOINCREMENT, event_id INTEGER REFERENCES event (id), event_attribute_name_id INTEGER REFERENCES event_attribute_name (id), value TEXT, type TEXT);

-- Table: classifier_attributes
CREATE TABLE IF NOT EXISTS classifier_attributes (id INTEGER PRIMARY KEY AUTOINCREMENT, classifier_id INTEGER REFERENCES classifier (id), event_attribute_name_id INTEGER REFERENCES event_attribute_name (id));

-- Table: object
CREATE TABLE IF NOT EXISTS object (id INTEGER PRIMARY KEY AUTOINCREMENT, class_id REFERENCES class (id));

-- Table: activity
CREATE TABLE IF NOT EXISTS activity (id INTEGER PRIMARY KEY AUTOINCREMENT, name TEXT);

--COMMIT TRANSACTION;
-- PRAGMA foreign_keys = on;

