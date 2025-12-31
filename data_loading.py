import pandas as pd
from sqlalchemy import create_engine



engine = create_engine(f"postgresql://{user}:{password}@{host}:{port}/{database}")

loan_train = pd.read_sql("SELECT * FROM loan_outcomes_train;", engine)
loan_predict = pd.read_sql("SELECT * FROM loan_outcomes_predict;", engine)
gps = pd.read_sql("SELECT * FROM gps;", engine)
events = pd.read_sql("SELECT * FROM events;", engine)
features = pd.read_sql("SELECT * FROM features;", engine)

loan_train.to_csv("data/loan_outcomes_train.csv", index=False)
loan_predict.to_csv("data/loan_outcomes_predict.csv", index=False)
gps.to_csv("data/gps.csv", index=False)
events.to_csv("data/events.csv", index=False)
features.to_csv("data/features.csv", index=False)