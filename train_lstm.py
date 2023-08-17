import sqlite3
import numpy as np
from model import ForexLSTMEncoder
import config

def create_dataset(X, time_steps=config.Config.SEQ_LEN):
    Xs = []
    for i in range(len(X) - time_steps):
        v = X[i:(i + time_steps)]
        Xs.append(v)
    return np.array(Xs)


def fetch_data_from_db(table_name, db_path="tradebot.db"):
    with sqlite3.connect(db_path) as conn:
        cursor = conn.cursor()
        query = f"SELECT * FROM {table_name}"
        cursor.execute(query)
        data = cursor.fetchall()
        column_names = [description[0] for description in cursor.description]
    return np.array(data)


def main():
    forex_model = ForexLSTMEncoder()

    tables = [
        "TrainingM1AUDUSD", "TrainingM1USDJPY", "TrainingM1GBPUSD",
        "TrainingM1USDCAD", "TrainingM1USDCHF", "TrainingM1EURUSD"
    ]

    all_data = []
    all_targets = []

    for table in tables:
        raw_data = fetch_data_from_db(table)
        raw_data_without_timestamp = raw_data[:, 1:]
        raw_data_without_timestamp = raw_data_without_timestamp.astype(float)

        data_features = np.hstack((
            raw_data_without_timestamp[:, 0:9],  # bid and ask ohlcv
            raw_data_without_timestamp[:, 10:]   # excluding volume and timestamp
        ))

        print(f"data_features Shape after hstack for {table}: {data_features.shape}")  # for Debugging only.

        targets = forex_model.get_targets(data_features)
        print(f"targets Shape after get_targets for {table}: {targets.shape}")  # for Debugging only.

        data_features = np.roll(data_features, shift=1, axis=0)[1:]
        targets = targets[:-1]

        print(f"data_features Shape after lookahead bias adjustment for {table}: {data_features.shape}")  # for Debugging only.
        print(f"targets Shape after lookahead bias adjustment for {table}: {targets.shape}")  # for Debugging only.

        all_data.append(create_dataset(data_features[:-1]))
        all_targets.append(targets[config.Config.TIMESTEPS-1:])

    all_data = np.concatenate(all_data, axis=0).astype(np.float32)
    all_targets = np.concatenate(all_targets, axis=0).astype(np.float32)


    forex_model.train_model(all_data, all_targets)

if __name__ == "__main__":
    main()
