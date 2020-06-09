import json
import os

with open("DBinfo.json") as f:
    database = json.load(f)


# multiline 주석 : shift + alt + A
class Setting():

    # DATA type
    DATATYPE: str = "deepfashion" # deepfashion / pirs

    # DB Auth info
    DATABASE_HOST: str = database['db']['host']
    DATABASE_USER: str = database['db']['user']
    DATABASE_PWD: str = database['db']['password']
    DATABASE_DB: str = database['db']['db']

    