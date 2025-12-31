from fastapi import FastAPI
from sqlalchemy import create_engine
import pandas as pd
import joblib

app = FastAPI(title="Loan Default Prediction API")

model = joblib.load("model_output/model.pkl")
imputer = joblib.load("model_output/imputer.pkl")
feature_cols = joblib.load("model_output/feature_cols.pkl")

DB_HOST = "branchhomeworkdb.cv8nj4hg6yra.ap-south-1.rds.amazonaws.com"
DB_PORT = "5432"
DB_USER = "datascientist"
DB_PASS = "47eyYBLT0laW5j9U24Uuy8gLcrN"
DB_NAME = "branchdsproject2025"

engine = create_engine(
    f"postgresql://{DB_USER}:{DB_PASS}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
)


@app.get("/")
def health_check():
    return {"message": "Loan Prediction API is running"}


@app.get("/predict")
def predict(user_id: int, application_at: str):
    """
    Predict repayment probability for a given user
    """


    loan_query = f"""
        SELECT *
        FROM loan_outcomes_predict
        WHERE user_id = {user_id}
        AND application_at = '{application_at}'
    """

    loan = pd.read_sql(loan_query, engine)

    if loan.empty:
        return {"error": "No matching user_id + application_at found"}

    features = pd.read_sql(
        f"SELECT * FROM features WHERE user_id = {user_id}",
        engine
    )

    df = loan.merge(features, on="user_id", how="left")

    application_time = pd.to_datetime(application_at)

    gps_query = f"""
        SELECT *
        FROM gps
        WHERE user_id = {user_id}
        AND time_of_fix <= '{application_at}'
    """

    gps = pd.read_sql(gps_query, engine)

    if not gps.empty:
        gps["time_of_fix"] = pd.to_datetime(gps["time_of_fix"])

        gps_features = (
            gps.groupby("user_id")
            .agg(
                gps_count=("id", "count"),
                avg_accuracy=("accuracy", "mean"),
                std_accuracy=("accuracy", "std"),
                avg_speed=("land_speed", "mean"),
                max_speed=("land_speed", "max"),
                providers=("location_provider", "nunique"),
                last_gps_time=("time_of_fix", "max"),
            )
            .reset_index()
        )

        gps_features["gps_recency_hours"] = (
            (application_time - gps_features["last_gps_time"])
            .dt.total_seconds()
            / 3600
        )

        gps_features.drop(columns=["last_gps_time"], inplace=True)

        df = df.merge(gps_features, on="user_id", how="left")

    events_query = f"""
        SELECT *
        FROM events
        WHERE user_id = {user_id}
        AND timestamp <= '{application_at}'
    """

    events = pd.read_sql(events_query, engine)

    if not events.empty:
        events["timestamp"] = pd.to_datetime(events["timestamp"])

        event_features = (
            events.groupby("user_id")
            .agg(
                total_events=("id", "count"),
                unique_screens=("screen_name", "nunique"),
                unique_actions=("action", "nunique"),
                total_sessions=("session_id", "nunique"),
                unique_networks=("network_type", "nunique"),
                last_event_time=("timestamp", "max"),
            )
            .reset_index()
        )

        event_features["event_recency_hours"] = (
            (application_time - event_features["last_event_time"])
            .dt.total_seconds()
            / 3600
        )

        event_features.drop(columns=["last_event_time"], inplace=True)

        df = df.merge(event_features, on="user_id", how="left")

    X = df[feature_cols]
    X = imputer.transform(X)

    probability = model.predict_proba(X)[0][1]

    return {
        "user_id": user_id,
        "application_at": application_at,
        "prediction_probability": float(probability),
    }
