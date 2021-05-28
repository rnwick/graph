# Scripts for O'Reilly's Graph Algorithm's
## Imports
import os.path as op
import pandas as pd
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)
data_path = ("../../data/")
from pyspark.sql.types import *
from graphframes import *

from graphframes.lib import AggregateMessages as AM
from pyspark.sql import functions as F

from operator import itemgetter



def new_paths(paths, id):
    paths = [{"id": col1, "distance": col2 + 1} for col1, col2 in paths if col1 != id]
    paths.append({"id": id, "distance": 1})
    return paths



def collect_paths(paths):
    return F.collect_set(paths)


def flatten(ids):
    return list(dict(sorted([item for sublist in ids for item in sublist], key=itemgetter(0))).items())


def merge_paths(ids, new_ids, id):
    return [{"id": col1, "distance": col2} for col1, col2 in dict(sorted([(col1, col2) for col1, col2 in ids + (new_ids if new_ids else []) if col1 != id], key=itemgetter(1), reverse=True)).items()]



def calculate_closeness(ids):
    return 0 if sum([col2 for col1, col2 in ids]) == 0 else len(ids) * 1.0 / sum([col2 for col1, col2 in ids])

