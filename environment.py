import sqlite3
from oandapyV20 import API
import oandapyV20.endpoints.accounts as accounts
import oandapyV20.endpoints.pricing as pricing
import oandapyV20.endpoints.trades as trades
from datetime import datetime
import time
from config import Config

class Environment:
    def __init__(self, access_token, account_id, instrument):
        self.access_token = Config.ACCESS_TOKEN
        self.account_id = Config.ACCOUNT_ID
        self.instrument = Config.INSTRUMENT
        self.client = API(access_token=access_token)
        self.conn = sqlite3.connect(Config.DB_NAME)
        self.cursor = self.conn.cursor()
        self.initial_balance = self.get_balance()
        self.session_active = True
        self.start_time = datetime.now()


        # Creating tables if not exists.
        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS prices (
                time TIMESTAMP,
                ask_open REAL,
                ask_close REAL,
                ask_high REAL,
                ask_low REAL,
                bid_open REAL,
                bid_close REAL,
                bid_high REAL,
                bid_low REAL,
                volume INTEGER
            )
        ''')

        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS trades (
                id INTEGER PRIMARY KEY,
                open_time TIMESTAMP,
                close_time TIMESTAMP,
                profit REAL
            )
        ''')

        self.conn.commit()

    def get_balance(self):
        try:
            r = accounts.AccountDetails(self.account_id)
            self.client.request(r)
            return float(r.response['account']['balance'])
        except Exception as e:
            print(f"An exception occurred: {e}")
            return None

    def calculate_position_size(self, stop_loss):
        risk_per_trade = self.balance * self.risk_per_trade + self.profit * self.risk_per_profit
        position_size = risk_per_trade / stop_loss
        return position_size


    def get_current_price(self):
        params = {"instruments": self.instrument}
        try:
            r = pricing.PricingInfo(accountID=self.account_id, params=params)
            rv = self.client.request(r)
            return float(rv["prices"][0]["bids"][0]["price"])
        except Exception as e:
            print(f"An exception occurred: {e}")
            return None

    def get_open_trades(self):
        try:
            r = trades.OpenTrades(accountID=self.account_id)
            self.client.request(r)
            return r.response['trades']
        except Exception as e:
            print(f"An exception occurred: {e}")
            return None

    def get_state(self):
        # Get the current open trades
        open_trades = self.get_open_trades()

        # Fetch the latest data
        self._get_data()

        # Calculate session time
        session_time = self._track_session_time()

        # Get the account balance
        balance = self.get_balance()

        # Create the state representation
        state = [self.data, open_trades, session_time, balance]

        return state


    def place_order(self, units, stop_loss_distance, take_profit_distance):
        """
        This function places an order on the Oanda platform.
        It automatically calculates and sets a stop loss and take profit level based on the provided distances.
        """

        try:
            # Fetch the current price
            price = self.get_current_price()

            # Calculate stop loss and take profit levels
            stop_loss_level = price - stop_loss_distance
            take_profit_level = price + take_profit_distance

            # Create the order request
            order_request = orders.OrderCreate(self.account_id, data={
                "order": {
                    "units": units,
                    "instrument": self.instrument,
                    "timeInForce": "FOK",
                    "type": "MARKET",
                    "positionFill": "DEFAULT_FILL",
                    "stopLossOnFill": {
                        "timeInForce": "GTC",
                        "price": str(stop_loss_level)
                    },
                    "takeProfitOnFill": {
                        "timeInForce": "GTC",
                        "price": str(take_profit_level)
                    }
                }
            })

            # Send the order request
            response = self.client.request(order_request)

            # Log the response
            self.log_response(response)

        except oandapyV20.exceptions.V20Error as err:
            # Log any errors that occur
            self.log_error(f"An error occurred while placing an order: {err}")

    # Close trades method
    def close_trade(self, trade_id):
        """
        This function closes an open trade on the Oanda platform.
        """

        try:
            # Create the trade close request
            trade_close_request = trades.TradeClose(self.account_id, trade_id)

            # Send the trade close request
            response = self.client.request(trade_close_request)

            # Log the response
            self.log_response(response)

        except oandapyV20.exceptions.V20Error as err:
            # Log any errors that occur
            self.log_error(f"An error occurred while closing a trade: {err}")

    # Reward calculation function
    def calculate_reward(self, action, open_trade):
        """
        This function calculates the reward for the agent based on the action taken and the result of the trade.
        """

        # Calculate profit or loss from closed trades
        reward = self.get_profit_or_loss()

        # If holding a losing trade for more than 15 minutes, subtract a small amount from the reward
        if open_trade and open_trade['profit'] < 0 and (datetime.now() - open_trade['open_time']).seconds > 900:
            reward -= 0.01

        # If no action is taken and not holding a position, subtract a very small amount from the reward
        if action == 'HOLD' and not open_trade:
            reward -= 0.001

        return reward

    # Logging functions
    def log_response(self, response):
        """
        This function logs the response from the Oanda API.
        """
        # For simplicity, we'll just print the response
        print(f"Response: {response}")

    def log_error(self, message):
        """
        This function logs any errors that occur.
        """
        # For simplicity, we'll just print the error message
        print(f"Error: {message}")

    # Ensure a 10% drawdown results in the system freezing for one hour
    def check_drawdown(self):
        """
        This function checks if a 10% drawdown on the account balance is reached.
        If yes, it initiates a 1-hour pause.
        """

        current_balance = self.get_balance()
        if current_balance < self.initial_balance * 0.9:
            print("10% drawdown reached. Trading will pause for 1 hour.")
            time.sleep(3600)  # pause for 1 hour

    def pause_session(self):
        self.session_active = False
        time.sleep(60*60)  # pauses for one hour
        self.session_active = True


    # Reset profit at beginning of each session
    def reset_profit(self):
        """
        This function resets the profit made in the last 2 hours at the start of each session.
        """
        current_time = datetime.datetime.now()
        two_hours_ago = current_time - datetime.timedelta(hours=2)

        # Update the trades table to reset the profit
        with sqlite3.connect(Config.DB_NAME) as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                UPDATE trades
                SET profit = 0
                WHERE open_time < ?
                """,
                (two_hours_ago,),
            )

    # Session tracking function
    def track_session_time(self):
        """
        This function tracks the session time.
        """

        current_time = datetime.now()
        elapsed_time = current_time - self.start_time
        session_time = elapsed_time.seconds

        return session_time

    def reset(self):
        self.reset_profit()
        self.strat_time = datetime.now()
        self._get_data()
# THIS FUNCTION NEEDS TO GET DATA FROM API IF IT DOES NOT ALREADY EXIST.
    # Load current data from db
    def _get_data(self):
        # Connect to SQLite database
        conn = sqlite3.connect(Config.DB_NAME)
        cur = conn.cursor()

        try:
            # Fetch the latest data from 'live_data' table
            cur.execute('SELECT * FROM live_data ORDER BY id DESC LIMIT 1')
            self.data = cur.fetchone()
        except sqlite3.Error as e:
            print(f"The error '{e}' occurred")
        finally:
            if conn:
                # Close the connection if it is open
                conn.close()
