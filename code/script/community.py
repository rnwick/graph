# Community


import os.path as op
import pandas as pd
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)
data_path = ("../../data/")
from pyspark.sql.types import *
from graphframes import *

from pyspark.sql import functions as F