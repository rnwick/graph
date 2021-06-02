from pyspark.sql.session import SparkSession
import os.path as op
import pandas as pd
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)

spark = SparkSession.builder.getOrCreate() 

data_path = ("../../data/")
rels_fname = op.join(data_path, 'transport-relationships.csv')
node_fname = op.join(data_path, 'transport-nodes.csv')
