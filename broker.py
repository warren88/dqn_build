import v20
import datetime
from config import Config
from data import initialize_database, count_data_points, insert_data_batch
from pprint import pprint

# Configuration
ctx = v20.Context(Config.API_URL, 443, token=Config.ACCESS_TOKEN)

# get most recent 1 minute candle data
def get_1min_candle(instrument):
    pass

# Place buy position
def place_buy_order(currency,amount,stop,profit): # buy instrument and set lot size, take profit and stop loss
    pass

# Place buy position
def place_sell_order(currency,amount,stop,profit):
    pass

# Place buy position
def close_position(trans_id): # close position
    pass

# Get most recent price data for live trading
def get_current_price(instrument):
    response = ctx.pricing.get(Config.ACCOUNT_ID, instruments=instrument)

    if response.status == 200:
        price_data = response.body["prices"][0]
        # Extracting data using dot notation
        bid = price_data.bids[0].price if price_data.bids else None
        ask = price_data.asks[0].price if price_data.asks else None
        instrument_name = price_data.instrument
        bid_liquidity = price_data.bids[0].liquidity if price_data.bids else None
        ask_liquidity = price_data.asks[0].liquidity if price_data.asks else None
        tradeable = price_data.tradeable
        quote_home_conversion_factors = price_data.quoteHomeConversionFactors

        return {
            "bid": bid,
            "ask": ask,
            "instrument": instrument_name,
            "bid_liquidity": bid_liquidity,
            "ask_liquidity": ask_liquidity,
            "tradeable": tradeable,
            "quoteHomeConversionFactors": quote_home_conversion_factors
        }

    else:
        error_message = response.body.get('errorMessage', 'Unknown error')
        print(f"Error fetching current price for {instrument}. Status: {response.status}. Error: {error_message}")
        return None


# Delete training tables, gather S15 and M1 data for previous 180 days and save in sqlite database
def fetch_training_data(pair, granularity):
    start_time = datetime.datetime.now()  # Start the timer before fetching data
    end_date = datetime.datetime.utcnow()
    start_date = end_date - datetime.timedelta(days=180)

    for start, end in get_date_ranges(start_date, end_date, granularity):
        print(f"Fetching {granularity} data for {pair} from {start} to {end}")

        response = ctx.instrument.candles(
            instrument=pair,
            price="BA",
            granularity=granularity,
            fromTime=start.isoformat() + "Z",
            toTime=end.isoformat() + "Z"
        )

        if response.status == 200:
            candles = response.body.get("candles", [])
            insert_data_batch(pair, candles, granularity)
        else:
            error_message = response.body.get('errorMessage', 'Unknown error')
            print(f"Error fetching data for {pair} from {start} to {end}. Status: {response.status}. Error: {error_message}")
    end_time = datetime.datetime.now()  # Stop the timer after fetching data
    elapsed_time = end_time - start_time  # Calculate elapsed time

    print(f"Time taken for {pair}: {elapsed_time}")

# Break down api calls in order to respect candle limit of 5000
def get_date_ranges(start_date, end_date, granularity="M1"):
    if granularity == "M1":
        delta = datetime.timedelta(days=3)
    elif granularity == "S15":
        delta = datetime.timedelta(seconds=75000)
    else:
        raise ValueError("Unsupported granularity")

    while start_date < end_date:
        next_date = start_date + delta
        yield start_date, next_date
        start_date = next_date

def main():
    initialize_database()
    overall_start_time = datetime.datetime.now()  # Start the overall timer
    for granularity in ["M1", "S15"]:
        for pair in Config.CURRENCY_PAIRS:
            print(f"Fetching {granularity} data for {pair}...")
            fetch_training_data(pair, granularity)
    overall_end_time = datetime.datetime.now()  # Stop the overall timer
    print("Data fetch and storage complete.")
    print(count_data_points())
    total_elapsed_time = overall_end_time - overall_start_time  # Calculate total elapsed time
    print(f"Total time taken: {total_elapsed_time}")



if __name__ == "__main__":
    instrument = "EUR_USD"
    data = get_current_price(instrument)

    if data:
        pprint(data)
    else:
        print(f"Failed to fetch price data for {instrument}")