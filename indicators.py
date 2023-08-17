import talib
from data import add_indicator_columns_to_table, get_close_prices, update_indicators_in_table
from config import Config
import sqlite3

def calculate_indicators_for_pair(granularity, pair):
    try:
        close_prices = get_close_prices(granularity, pair)

        # Calculate RSI
        rsi_period = 14 if len(close_prices) >= 14 else len(close_prices)
        rsi = talib.RSI(close_prices, timeperiod=rsi_period)

        # Bollinger Bands
        bb_period = 20 if len(close_prices) >= 20 else len(close_prices)
        upper, middle, lower = talib.BBANDS(close_prices, timeperiod=bb_period)

        # MACD
        macd, macd_signal, macd_hist = talib.MACD(close_prices, fastperiod=12, slowperiod=26, signalperiod=9)

        indicators = {
            'close_prices': close_prices,
            'rsi': rsi,
            'upper': upper,
            'middle': middle,
            'lower': lower,
            'macd': macd,
            'macd_signal': macd_signal,
            'macd_hist': macd_hist
        }

        update_indicators_in_table(granularity, pair, indicators)
        print(f"Successfully calculated indicators for {pair} with {granularity} granularity.")

    except Exception as e:
        print(f"Error while calculating indicators for {pair} with {granularity} granularity: {str(e)}")

# ensure indicators are filled, if data not available fill first first available value
def backfill_missing_indicator_values(granularity, pair):
    table_name = "Training" + granularity + pair.replace('_', '')

    total_backfilled = 0

    with sqlite3.connect(Config.DB_NAME) as conn:
        cursor = conn.cursor()

        # Handle RSI
        try:
            cursor.execute(f"SELECT rowid FROM {table_name} WHERE RSI IS NULL")
            missing_rsi_rows = [row[0] for row in cursor.fetchall()]

            if missing_rsi_rows:
                cursor.execute(f"SELECT RSI FROM {table_name} WHERE RSI IS NOT NULL LIMIT 1")
                first_rsi_value = cursor.fetchone()[0]

                for row_id in missing_rsi_rows:
                    cursor.execute(f"UPDATE {table_name} SET RSI=? WHERE rowid=?", (first_rsi_value, row_id))
                print(f"Backfilled {len(missing_rsi_rows)} missing RSI values.")
                total_backfilled += len(missing_rsi_rows)
        except Exception as e:
            print(f"Error backfilling RSI for {pair} with {granularity} granularity: {str(e)}")

        # Handle Bollinger Bands
        try:
            cursor.execute(f"SELECT rowid FROM {table_name} WHERE BB_UPPER IS NULL OR BB_MIDDLE IS NULL OR BB_LOWER IS NULL")
            missing_bb_rows = [row[0] for row in cursor.fetchall()]

            if missing_bb_rows:
                cursor.execute(f"SELECT BB_UPPER, BB_MIDDLE, BB_LOWER FROM {table_name} WHERE BB_UPPER IS NOT NULL LIMIT 1")
                first_upper, first_middle, first_lower = cursor.fetchone()

                for row_id in missing_bb_rows:
                    cursor.execute(f"UPDATE {table_name} SET BB_UPPER=?, BB_MIDDLE=?, BB_LOWER=? WHERE rowid=?",
                                (first_upper, first_middle, first_lower, row_id))
                print(f"Backfilled {len(missing_bb_rows)} missing Bollinger Bands values.")
                total_backfilled += len(missing_bb_rows)
        except Exception as e:
            print(f"Error backfilling Bollinger Bands for {pair} with {granularity} granularity: {str(e)}")

        # Handle MACD
        try:
            cursor.execute(f"SELECT rowid FROM {table_name} WHERE MACD IS NULL OR MACD_SIGNAL IS NULL OR MACD_HIST IS NULL")
            missing_macd_rows = [row[0] for row in cursor.fetchall()]

            if missing_macd_rows:
                cursor.execute(f"SELECT MACD, MACD_SIGNAL, MACD_HIST FROM {table_name} WHERE MACD IS NOT NULL LIMIT 1")
                first_macd, first_macd_signal, first_macd_hist = cursor.fetchone()

                for row_id in missing_macd_rows:
                    cursor.execute(f"UPDATE {table_name} SET MACD=?, MACD_SIGNAL=?, MACD_HIST=? WHERE rowid=?",
                                (first_macd, first_macd_signal, first_macd_hist, row_id))
                print(f"Backfilled {len(missing_macd_rows)} missing MACD values.")
                total_backfilled += len(missing_macd_rows)
        except Exception as e:
            print(f"Error backfilling MACD for {pair} with {granularity} granularity: {str(e)}")

        conn.commit()

    print(f"Total backfilled values for {pair} with {granularity} granularity: {total_backfilled}")

# You can then call this function after you've calculated the indicators in your main script:
# backfill_missing_indicator_values(granularity, pair)

if __name__ == "__main__":
    for granularity in ['M1', 'S15']:
        for pair in Config.CURRENCY_PAIRS:
            print(f"Processing {pair} with {granularity} granularity...")
            try:
                add_indicator_columns_to_table(granularity, pair)
                calculate_indicators_for_pair(granularity, pair)

                # Call the backfill function here, after indicators are calculated
                backfill_missing_indicator_values(granularity, pair)

            except Exception as e:
                print(f"Error while processing {pair} with {granularity} granularity: {str(e)}")

    print("Indicators calculation and backfilling process completed.")
