import omidb

OMIDB_DATA_PATH = '/Users/hendrik/Studium/Master/Thesis/Data/OMI-DB Sample/DATA'
OMIDB_IMAGES_PATH = '/Users/hendrik/Studium/Master/Thesis/Data/OMI-DB Sample/IMAGES'
db = omidb.DB(OMIDB_DATA_PATH, OMIDB_IMAGES_PATH)

print(db)
