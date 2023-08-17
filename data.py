import sqlite3
from config import Config
import numpy as np

DATABASE_NAME = Config.DB_NAME

INSERT_INTO_TABLE_QUERY = """
INSERT INTO {table_name} (
    timestamp, bid_open, bid_high, bid_low, bid_close,
    ask_open, ask_high, ask_low, ask_close, volume
) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
"""

# Initialise database.
def initialize_database():
    conn = sqlite3.connect(DATABASE_NAME)
    cursor = conn.cursor()

    for granularity in ["M1", "S15"]:
        # Drop training tables to ensure the table only contains current data
        for pair in Config.CURRENCY_PAIRS:
            table_name = "Training" + granularity + pair.replace('_', '')
            cursor.execute(f"DROP TABLE IF EXISTS {table_name}")

        # Create tables for training data
        for pair in Config.CURRENCY_PAIRS:
            table_name = "Training" + granularity + pair.replace('_', '')
            cursor.execute(f"""
            CREATE TABLE IF NOT EXISTS {table_name} (
                timestamp TEXT NOT NULL,
                bid_open REAL NOT NULL,
                bid_high REAL NOT NULL,
                bid_low REAL NOT NULL,
                bid_close REAL NOT NULL,
                ask_open REAL NOT NULL,
                ask_high REAL NOT NULL,
                ask_low REAL NOT NULL,
                ask_close REAL NOT NULL,
                volume INTEGER NOT NULL
            )
            """)

    conn.commit()
    conn.close()

# Add columns to save indicator values
def add_indicator_columns_to_table(granularity, pair):
    table_name = "Training" + granularity + pair.replace('_', '')

    # A list of the columns to add and their data types
    columns_to_add = {
        "RSI": "REAL",
        "BB_UPPER": "REAL",
        "BB_MIDDLE": "REAL",
        "BB_LOWER": "REAL",
        "MACD": "REAL",
        "MACD_SIGNAL": "REAL",
        "MACD_HIST": "REAL"
    }

    with sqlite3.connect(DATABASE_NAME) as conn:
        cursor = conn.cursor()

        # Fetch existing columns
        cursor.execute(f"PRAGMA table_info({table_name})")
        existing_columns = [column[1] for column in cursor.fetchall()]

        # Add new columns for the indicators
        for column_name, data_type in columns_to_add.items():
            if column_name not in existing_columns:
                cursor.execute(f"ALTER TABLE {table_name} ADD COLUMN {column_name} {data_type}")

        conn.commit()

# Add indicator values to table data
def update_indicators_in_table(granularity, pair, indicators):
    table_name = "Training" + granularity + pair.replace('_', '')

    with sqlite3.connect(DATABASE_NAME) as conn:
        cursor = conn.cursor()
        for i in range(len(indicators['close_prices'])):
            cursor.execute(f"""
            UPDATE {table_name}
            SET RSI=?, BB_UPPER=?, BB_MIDDLE=?, BB_LOWER=?, MACD=?, MACD_SIGNAL=?, MACD_HIST=?
            WHERE rowid=?
            """, (
                indicators['rsi'][i],
                indicators['upper'][i],
                indicators['middle'][i],
                indicators['lower'][i],
                indicators['macd'][i],
                indicators['macd_signal'][i],
                indicators['macd_hist'][i],
                i + 1
            ))

        conn.commit()

# Required for adding indicators
def get_close_prices(granularity, pair):
    table_name = "Training" + granularity + pair.replace('_', '')

    with sqlite3.connect(DATABASE_NAME) as conn:
        cursor = conn.cursor()
        cursor.execute(f"SELECT bid_close, ask_close FROM {table_name}")
        data = cursor.fetchall()  # Fetch both bid_close and ask_close prices.

        # Calculate the average of bid_close and ask_close for each row.
        avg_close_prices = [(item[0] + item[1]) / 2 for item in data]

        return np.asarray(avg_close_prices)
"""
# Get all M1 data from tables for training lstm
def get_m1_data():
    table_name = "Training" + granularity + pair.replace('_', '')

    with sqlite3.connect(DATABASE_NAME) as conn:
        cursor = conn.cursor()
        cursor.execute(f"SELECT * FROM {table_name}")
        data = cursor.fetchall()  # Fetch both bid_close and ask_close prices.

        # Calculate the average of bid_close and ask_close for each row.
        avg_close_prices = [(item[0] + item[1]) / 2 for item in data]

        return np.asarray(avg_close_prices)
"""
# Check training data is stored, report number of data points.
def count_data_points():
    conn = sqlite3.connect(DATABASE_NAME)
    cursor = conn.cursor()

    total_count = 0

    for granularity in ["M1", "S15"]:
        for pair in Config.CURRENCY_PAIRS:
            table_name = "Training" + granularity + pair.replace('_', '')
            cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
            count = cursor.fetchone()[0]  # fetchone() returns a tuple, we want the first value.
            total_count += count
            print(f"{table_name} contains {count} data points.")
    conn.close()
    print(f"Total data points across all tables: {total_count}")

def insert_data_batch(pair, candles, granularity):
    conn = sqlite3.connect(DATABASE_NAME)
    cursor = conn.cursor()

    for candle in candles:
        cursor.execute(INSERT_INTO_TABLE_QUERY.format(table_name="Training" + granularity + pair.replace('_', '')), (
            candle.time,
            candle.bid.o, candle.bid.h, candle.bid.l, candle.bid.c,
            candle.ask.o, candle.ask.h, candle.ask.l, candle.ask.c,
            candle.volume
        ))
    conn.commit()
    conn.close()

if __name__ == "__main__":
    count_data_points()