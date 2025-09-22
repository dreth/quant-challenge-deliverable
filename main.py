from __future__ import annotations

import logging as log
import os
from decimal import Decimal

import databento as db
import matplotlib.pyplot as plt
import pandas as pd
from dotenv import load_dotenv
from nautilus_trader.backtest.config import BacktestEngineConfig
from nautilus_trader.backtest.engine import BacktestEngine
from nautilus_trader.config import StrategyConfig
from nautilus_trader.indicators.averages import SimpleMovingAverage
from nautilus_trader.model import (
    BarType,
    InstrumentId,
    Money,
    Price,
    Quantity,
    Symbol,
    TraderId,
    Venue,
)
from nautilus_trader.model.currencies import USD
from nautilus_trader.model.data import Bar
from nautilus_trader.model.enums import (
    AccountType,
    AssetClass,
    OmsType,
    OrderSide,
)
from nautilus_trader.model.instruments import FuturesContract
from nautilus_trader.trading.strategy import Strategy

# List of E-mini S&P 500 futures contracts to query.
contracts = [
    "ESH4",  # March 2024
    "ESM4",  # June 2024
    "ESU4",  # September 2024
    "ESZ4",  # December 2024
    "ESH5",  # March 2025
    "ESM5",  # June 2025
    "ESU5",  # September 2025
    "ESZ5",  # December 2025
]


def query_data_from_databento() -> pd.DataFrame:
    """Query E-mini S&P 500 futures data from Databento."""
    # list to hold dataframes
    dfs = []

    # Make sure to set your DATABENTO_API_KEY in a .env file or environment variable
    if not os.getenv("DATABENTO_API_KEY"):
        raise ValueError("DATABENTO_API_KEY environment variable not set.")

    # Databento Client
    client = db.Historical(os.getenv("DATABENTO_API_KEY"))

    # Iterate over contracts and query data
    for contract in contracts:
        data = client.timeseries.get_range(
            dataset="GLBX.MDP3",  # E-mini S&P 500 Futures
            symbols=contract,  # Contract to query
            schema="ohlcv-1m",  # 1 minute OHLCV bars
            start="2023-12-31T23:59:00",  # start of the year 2024
            end="2025-01-01T00:00:00",  # end of the year 2024
        )  # This returns a DBNStore object which has the methods to_df() and to_csv()

        # return a CSV file per contract into the contracts dir
        data.to_csv(os.path.join("data", "contracts", f"data_{contract}_2024.csv"))

        # Convert the data to a pandas Dataframe to stitch them together later
        dfs.append(data.to_df())

        # print which contract we're querying
        log.info(f"Queried {contract}")

    # create the raw_data df
    raw_data = pd.concat(dfs).reset_index()

    # save the concatenated dataframe to a CSV file
    raw_data.to_csv(os.path.join("data", "queried_2024_data.csv"))

    return raw_data


# i dont like how this script looks, I would have made it more modular
# and maybe build a cool CLI around it with Typer or Click, but time is limited and i refuse to use ai for it
def obtain_and_process_data() -> pd.DataFrame:
    """Main function to query, process, and stitch E-mini S&P 500 futures data."""

    # if the final data is present, no need to do anything just continue with the strategy
    if not os.path.exists(os.path.join("data", "final_data.csv")):

        # Query timeseries data per contract
        if not os.path.exists(os.path.join("data", "queried_2024_data.csv")):

            # query the data
            raw_data = query_data_from_databento()

        # Load the data if already queried
        else:
            raw_data = pd.read_csv(
                os.path.join("data", "queried_2024_data.csv")
            ).reset_index(drop=True)
            log.info("Loaded queried data from CSV")

        # remove spread contracts
        data = raw_data.copy()
        data = data[~data["symbol"].str.contains("-")]

        # rename cols
        data.rename(columns={"ts_event": "ts", "symbol": "contract"}, inplace=True)

        # parse timestamps
        data["ts"] = pd.to_datetime(data["ts"], utc=True)

        # sort values by ts and contract
        data.sort_values(by=["ts", "contract"], inplace=True)
        data.set_index("ts", inplace=True)

        # make a copy of 'data' for manip.
        df = data.copy()

        # calculate rolling 3D volume per contract
        roll_vol = (
            df.groupby("contract")["volume"]
            .rolling("3D")
            .sum()
            .reset_index()
            .set_index(["ts", "contract"])["volume"]
        )

        # set ts and contract as the index
        data.reset_index(inplace=True)
        data.set_index(["ts", "contract"], inplace=True)
        data["rolling_volume"] = roll_vol

        # fill NAs in rolling_volume with 0
        data.fillna({"rolling_volume": 0}, inplace=True)

        # mark the 'higher_vol' row for the highest rolling_volume per ts and contract
        winner_idx = data.groupby(level=0)["rolling_volume"].idxmax()
        data["higher_vol"] = False
        data.loc[winner_idx, "higher_vol"] = True

        # mark the rows where a contract change has occurred
        df = data.copy()
        df = df.loc[winner_idx].reset_index().sort_values(by=["ts", "contract"])
        df = df.reset_index().set_index("ts")
        df["roll_occurred"] = df["contract"].ne(df["contract"].shift())
        df.loc[df.index[0], "roll_occurred"] = False  # first row is not a roll

        # get the expiry date of each contract
        df_for_expiry = data.reset_index()[["ts", "contract"]]
        contract_expiry = {
            contract: df_for_expiry[df_for_expiry["contract"] == contract].ts.max()
            for contract in contracts
        }

        # create a mapping of contract to its order
        contract_order = {x: n for n, x in enumerate(contracts)}
        roll_events = df[df["roll_occurred"]].assign(
            contract_order=lambda d: d["contract"].map(contract_order)
        )

        # log how many potential roll events there could be
        log.info(f"Potential roll events found: {len(roll_events)}")

        # we want to avoid rolling back into a previous contract
        contracts_passed = set(["ESH4"])
        contract_order_passed = set([0])

        # iterate over contracts, if a roll occurs into a contract in roll_events, we add it to the contracts_passed set and then any contract _previous_ to this one in the order, is removed from the roll_events table
        rolls = []
        for idx, roll in roll_events.iterrows():
            # if the contract has never been rolled into before
            if roll["contract"] not in contracts_passed:
                # let's avoid taking a roll that's too early before expiry
                previous_contract = contracts[contracts.index(roll["contract"]) - 1]
                if (contract_expiry[previous_contract] - roll.name) <= pd.Timedelta(
                    "2W"
                ):
                    # log info
                    log.info(f"Rolling into {roll['contract']} at {roll.name}")

                    # and if no previous contract in the order has been rolled into before
                    if any(
                        [x >= roll["contract_order"] for x in contract_order_passed]
                    ):
                        pass
                    else:
                        # we save this roll event so that we won't roll back into the previous contract
                        contracts_passed.add(roll["contract"])
                        contract_order_passed.add(roll["contract_order"])

                        # save the rolls to iterate over them later
                        rolls.append((idx, roll["contract"]))

                else:
                    log.info(
                        f"Skipping roll into {roll['contract']} at {roll.name} as it's too early before expiry"
                    )

        # mark the rolls in the main dataset
        # list to concatenate chunks of data
        data_to_append = []

        # overlap the data to apply Panama method
        overlap = pd.Timedelta("7D")

        # reset index to filter by ts and contract
        data.reset_index(inplace=True)

        # add the first chunk of data before the first roll
        # before the first timestamp we take the first contract
        data_to_append.append(
            data[(data["ts"] < rolls[0][0]) & (data["contract"] == contracts[0])].copy()
        )

        # loop through timestamp and contract where each contract roll occurs
        for i, (timestamp, contract) in enumerate(rolls):

            # logging info
            log.info(f"Processing roll into {contract} at {timestamp}")

            # next timestamp
            next_timestamp = (
                rolls[i + 1][0]
                if i + 1 < len(rolls)
                else pd.Timestamp("2025-01-01T00:00:00", tz="UTC")
            )

            # obtain the data for the current contract from previous roll to current roll
            temp_df = data[
                (data["ts"] >= (timestamp - overlap))
                & (data["ts"] < next_timestamp)
                & (data["contract"] == contract)
            ].copy()

            # append to list
            data_to_append.append(temp_df)

        # concatenate all the dataframes in the list
        appended_data = pd.concat(data_to_append)

        # drop 'higher_vol' column and other useless cols
        appended_data.drop(
            columns=[
                "higher_vol",
                "publisher_id",
                "rtype",
                "rolling_volume",
                "instrument_id",
            ],
            inplace=True,
        )

        # make 'contract' a categorical variable
        appended_data["contract"] = pd.Categorical(
            appended_data["contract"], categories=contracts, ordered=True
        )

        # order appended_data by ts and contract
        appended_data.sort_values(by=["ts", "contract"], inplace=True)

        # overlap dataframe
        overlap = appended_data[
            appended_data.duplicated(subset=["ts"], keep=False)
        ].copy()

        # panama method
        cols_to_back_adjust = ["close", "open", "high", "low"]
        overlap_cp = overlap.copy()
        for col in cols_to_back_adjust:
            # shift the next contract up to subtract from previous contract
            overlap_cp[f"{col}_next"] = overlap_cp.groupby("ts")["close"].shift(-1)
            overlap_cp[f"{col}_diff"] = overlap_cp["close"] - overlap_cp["close_next"]

        # collect indeces of the overlapping futures contracts that we will want to drop from the main data
        overlap_to_remove = overlap_cp[
            (overlap_cp["open_next"].isna())
            | (overlap_cp["high_next"].isna())
            | (overlap_cp["low_next"].isna())
            | (overlap_cp["close_next"].isna())
        ].index

        # drop all the 'future' contracts in the overlap_cp df
        for col in cols_to_back_adjust:
            overlap_cp.dropna(subset=[f"{col}_next"], inplace=True)

        # get the median difference per contract per column
        overlap_median_diffs = overlap_cp.groupby("contract", observed=True)[
            [f"{col}_diff" for col in cols_to_back_adjust]
        ].median()

        # remove the rows of the overlapping contracts that we don't want, only keeping the previous contract prior to the window where both contracts exist
        appended_data = appended_data[~appended_data.index.isin(overlap_to_remove)]

        # apply the back adjustment to the data
        for contract in overlap_median_diffs.index:

            # apply the diff to all rows of the contract
            for col in cols_to_back_adjust:

                # get the diff
                diff = overlap_median_diffs.loc[contract, f"{col}_diff"]

                # apply the diff only to numeric columns
                mask = appended_data["contract"] == contract
                appended_data.loc[mask, col] = appended_data.loc[mask, col].astype(float) - diff  # type: ignore

        # finally order by timestamp (contract should be ordered at this point without specifying it explicitly)
        appended_data.sort_values(by="ts", inplace=True)

        # save the final data to CSV
        appended_data.to_csv(os.path.join("data", "final_data.csv"), index=False)

        # generate a simple plot with mpl with the data (just for confirmation)
        plt.figure(figsize=(12, 6))
        plt.plot(appended_data["ts"], appended_data["close"], linewidth=1)
        plt.xlabel("Time")
        plt.ylabel("Closing Price")
        plt.title("E-mini S&P 500 Futures - Continuous Contract (Back-Adjusted)")
        plt.xticks(rotation=45)
        plt.tight_layout()

        # save as an image
        plt.savefig(
            os.path.join("artifacts", "closing_price_plot.png"),
            dpi=300,
            bbox_inches="tight",
        )
        plt.close()
        log.info("Plot saved to artifacts/closing_price_plot.png")

    else:
        # load the final data if it already exists
        appended_data = pd.read_csv(os.path.join("data", "final_data.csv"))
        log.info("Loaded final data from CSV")

    # process data for backtesting
    df = appended_data.copy()

    # validate that we have the required timestamp column
    # the data should contain: ts, open, high, low, close, [volume]
    if "ts" not in df.columns:
        raise ValueError("final_data.csv must contain 'ts' (UTC ISO timestamps).")

    # rename timestamp column to match expectations and convert to datetime
    df = df.rename(columns={"ts": "timestamp"})
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)

    # clean and convert price data to proper numeric types
    # nautilus is very strict about data types
    for c in ["open", "high", "low", "close"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # handle volume data (set to 0 if missing)
    if "volume" not in df.columns:
        df["volume"] = 0
    df["volume"] = (
        pd.to_numeric(df["volume"], errors="coerce").fillna(0).astype("int64")
    )

    # remove any rows with missing price data and sort by time
    df = (
        df.dropna(subset=["open", "high", "low", "close"])
        .set_index("timestamp")
        .sort_index()
    )

    # ensure we have all required columns with correct data types
    # nautilus is very strict about data types and format
    needed = ["open", "high", "low", "close", "volume"]
    for col in needed:
        if col not in df.columns:
            raise ValueError(f"final_data.csv missing required column '{col}'")
    df_use = df[needed].copy()

    # explicitly cast to the exact types nautilus expects
    df_use["open"] = df_use["open"].astype("float64")
    df_use["high"] = df_use["high"].astype("float64")
    df_use["low"] = df_use["low"].astype("float64")
    df_use["close"] = df_use["close"].astype("float64")
    df_use["volume"] = df_use["volume"].astype("int64")

    # return the processed dataframe
    return df_use


def setup_bars(df: pd.DataFrame) -> tuple[BarType, list[Bar]]:
    """setup bars from a pandas dataframe

    Args:
        df (pd.DataFrame): dataframe with ohlcv data and datetime index

    Returns:
        tuple[BarType, list[Bar]]: bartype and list of bar objects
    """
    # define bar type
    # 1-minute bars using last (close) prices from external data source
    bar_type = BarType.from_str("ES.CONT.XCME-1-MINUTE-LAST-EXTERNAL")

    # bars list to hold the converted bar objects
    bars = []
    for ts, row in df.iterrows():
        # convert pandas timestamp to nanoseconds (nautilus internal format)
        t_ns = int(ts.value)  # type: ignore

        # create bar object with all ohlcv data
        bars.append(
            Bar.from_dict(
                {
                    "bar_type": str(bar_type),  # which instrument/timeframe
                    "open": f"{row['open']:.2f}",  # opening price
                    "high": f"{row['high']:.2f}",  # highest price
                    "low": f"{row['low']:.2f}",  # lowest price
                    "close": f"{row['close']:.2f}",  # closing price
                    "volume": str(int(row["volume"])),  # volume traded
                    "ts_event": t_ns,  # when the bar was formed
                    "ts_init": t_ns,  # when we received the data
                }
            )
        )

    return bar_type, bars


def generate_reports(venue: Venue, engine: BacktestEngine, data: pd.DataFrame) -> None:
    """Generate various reports from the backtest engine and save them to the specified path.

    Args:
        venue (Venue): venue where the instrument trades
        engine (BacktestEngine): backtest engine instance
        data (pd.DataFrame): dataframe with historical price data
    """
    reports_path = "reports"
    if not os.path.exists(reports_path):
        os.makedirs(reports_path)

    # account balance, pnl, drawdown
    account_report = engine.trader.generate_account_report(venue)
    account_report.to_csv(os.path.join(reports_path, "account_report.csv"))

    # position history and stats
    engine.trader.generate_positions_report().to_csv(
        os.path.join(reports_path, "positions_report.csv")
    )

    # all trades executed
    engine.trader.generate_order_fills_report().to_csv(
        os.path.join(reports_path, "fills_report.csv")
    )

    # collect comprehensive results
    stats_pnls = engine.portfolio.analyzer.get_performance_stats_pnls()
    stats_returns = engine.portfolio.analyzer.get_performance_stats_returns()
    stats_general = engine.portfolio.analyzer.get_performance_stats_general()

    # comparison with buy and hold and close price in first timestamp
    total_investment = 100000
    buy_and_hold = (total_investment / float(data["close"].iloc[0])) * float(
        data["close"].iloc[-1]
    )
    buy_and_hold_pnl = {"Buy and Hold PnL": f"${buy_and_hold:.2f}"}

    # buy and hold ROI
    buy_and_hold_roi = ((buy_and_hold / total_investment) * 100) - 100
    buy_and_hold_roi_stat = {"Buy and Hold ROI": f"{buy_and_hold_roi:.2f}%"}

    # comparison with monthly DCA strategy
    # invest $8333.33 at the close price on the first trading day of each
    # month over the backtest period
    monthly_investment = total_investment / 12
    dca_investment = 0.0
    dca_shares = 0.0
    current_month = None
    for ts, row in data.iterrows():
        # we know ts is a timestamp, so let's ignore the typechecker here
        if ts.month != current_month:  # type: ignore
            current_month = ts.month
            dca_investment += monthly_investment
            dca_shares += monthly_investment / row["close"]
    dca_final_value = dca_shares * data["close"].iloc[-1]
    dca_roi = ((dca_final_value / dca_investment) * 100) - 100
    dca_stats = {
        "DCA Investment": f"${dca_investment:.2f}",
        "DCA Final Value": f"${dca_final_value:.2f}",
        "DCA ROI": f"{dca_roi:.2f}%",
    }

    # create summary dictionary
    results = (
        stats_pnls
        | stats_returns
        | stats_general
        | buy_and_hold_pnl
        | buy_and_hold_roi_stat
        | dca_stats
    )
    results = pd.DataFrame(
        {
            "Statistic": results.keys(),
            "Value": results.values(),
        }
    )

    # return summary results as a markdown table in an .md file
    results.to_markdown(os.path.join(reports_path, "summary.md"), index=False)

    # generate a plot of account balance over time
    _, ax = plt.subplots(figsize=(20, 10))
    ax.plot(
        pd.to_datetime(account_report.index),
        account_report["total"].astype(float),
        label="Account Balance",
        color="blue",
        linewidth=1,
    )
    ax.locator_params(axis="y", nbins=8)
    ax.set_xlabel("Time")
    ax.set_ylabel("Balance (USD)")
    ax.set_title("Account Balance Over Time")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(reports_path, "account_balance.png"))
    plt.close()


def setup_instrument() -> FuturesContract:
    """
    define and return the futurescontract object for es continuous futures.
    """
    # create a synthetic continuous contract id for es futures
    # this represents a "rolled" contract that doesn't expire
    instrument_id = InstrumentId.from_str("ES.CONT.XCME")

    # define the futures contract specifications
    # es (e-mini s&p 500) contract details:
    instrument = FuturesContract(
        instrument_id=instrument_id,  # unique identifier: instrumentid
        raw_symbol=Symbol("ES_CONT"),  # trading symbol: symbol
        asset_class=AssetClass.INDEX,  # asset class (equity index): assetclass
        underlying="S&P 500",  # underlying asset description: str
        activation_ns=0,  # activation time (0 = active now): int
        expiration_ns=2**63 - 1,  # expiration time (far future = never expires): int
        currency=USD,  # base currency: currency
        price_precision=2,  # price precision (2 decimal places): int
        price_increment=Price.from_str("0.25"),  # minimum tick size ($0.25): price
        multiplier=Quantity.from_int(50),  # point value ($50 per point): quantity
        lot_size=Quantity.from_int(1),  # minimum lot size (1 contract): quantity
        ts_event=0,  # event timestamp: int
        ts_init=0,  # init timestamp: int
    )
    return instrument


def venue() -> Venue:
    """
    create and return the trading venue (exchange) where the instrument trades.
    """
    return Venue("XCME")  # chicago mercantile exchange


def backtest(df: pd.DataFrame) -> None:
    """
    main backtesting function that runs the sma crossover strategy on historical data.
    """
    # create the venue (exchange) where the instrument trades
    CME = venue()

    # --- setup instrument and data ---
    # define the futures contract specifications
    # es (e-mini s&p 500) contract details:
    log.info("setting up trading instrument...")
    instrument = setup_instrument()

    # preprocess the historical price data into nautilus bar objects
    log.info("converting price data to nautilus bar objects...")
    bar_type, bars = setup_bars(df)
    log.info(f"created {len(bars)} bar objects for backtesting")

    # --- create backtest engine ---
    # create the backtest engine with a unique trader id
    log.info("initializing backtest engine...")
    engine = BacktestEngine(config=BacktestEngineConfig(trader_id=TraderId("BT-1")))

    # add the trading venue with account configuration
    engine.add_venue(
        venue=CME,  # chicago mercantile exchange
        oms_type=OmsType.NETTING,  # order management system type (net positions)
        account_type=AccountType.MARGIN,  # margin account (required for futures)
        base_currency=USD,  # account base currency
        starting_balances=[Money(100_000, USD)],  # starting capital: $100,000
    )

    # register our es futures contract with the engine
    engine.add_instrument(instrument)

    # load all the historical price data into the engine
    engine.add_data(bars)
    log.info(f"loaded {len(bars)} bars into backtest engine")

    # --- configure and add strategy ---
    # create strategy configuration with our parameters
    log.info("setting up sma crossover strategy...")
    cfg = SMACrossConfig(
        instrument_id=instrument.id,  # trade the es continuous contract
        bar_type=bar_type,  # use 1-minute bars
        fast=20,  # 20-period fast moving average
        slow=50,  # 50-period slow moving average
        trade_qty=1,  # trade 1 contract per signal
    )

    # create and add the strategy instance to the engine
    engine.add_strategy(SMACrossStrategy(cfg))

    # --- run backtest ---
    # execute the backtest - this runs through all historical data
    # and simulates trading based on the strategy logic
    engine.run()

    log.info("backtest completed! generating reports...")

    # --- reporting ---
    generate_reports(CME, engine, df.copy())

    # clean up resources
    engine.dispose()

    log.info("Check the console output above for detailed performance metrics.")


# config object
# frozen so nautilus can validate/hash it cleanly
class SMACrossConfig(StrategyConfig, frozen=True):
    instrument_id: InstrumentId
    bar_type: BarType
    fast: int = 20  # default = 20 period sma
    slow: int = 50  # default = 50 period sma
    trade_qty: int = 1  # default trade size = 1 contract/share


class SMACrossStrategy(Strategy):
    """
    simple sma crossover
    - long when fast > slow
    - flat when fast < slow
    """

    def __init__(self, config: SMACrossConfig) -> None:
        super().__init__(config)

        # indicators we care about
        self.fast = SimpleMovingAverage(period=config.fast)
        self.slow = SimpleMovingAverage(period=config.slow)

        # cache instrument object (tick size, contract specs, etc.)
        self.instrument = None

    def on_start(self) -> None:
        # cache instrument once
        self.instrument = self.cache.instrument(self.config.instrument_id)

        # register indicators so they auto-update when new bars come in
        self.register_indicator_for_bars(self.config.bar_type, self.fast)
        self.register_indicator_for_bars(self.config.bar_type, self.slow)

        # subscribe to the bar stream
        # backtests will automatically feed history, so no need for request_bars here
        self.subscribe_bars(self.config.bar_type)

    def on_bar(self, bar: Bar) -> None:
        # wait until both indicators are warmed up
        if not (self.fast.initialized and self.slow.initialized):
            return

        # check if we're currently holding a long
        is_long = self.portfolio.is_net_long(self.config.instrument_id)

        # entry condition: if flat and fast > slow -> buy
        if (not is_long) and (self.fast.value > self.slow.value):
            self._enter_long()
            return

        # exit condition: if long and fast < slow -> close position
        if is_long and (self.fast.value < self.slow.value):
            self._flatten()

    # helper: submit a long market order
    def _enter_long(self) -> None:
        # object caching
        self.instrument = self.cache.instrument(self.config.instrument_id)

        # create and submit order
        qty = self.instrument.make_qty(Decimal(self.config.trade_qty))
        order = self.order_factory.market(
            instrument_id=self.config.instrument_id,
            order_side=OrderSide.BUY,
            quantity=qty,
        )
        self.submit_order(order)

    # helper: flatten (close existing positions only)
    def _flatten(self) -> None:
        self.close_all_positions(self.config.instrument_id)


if __name__ == "__main__":
    # create required dirs
    os.makedirs("data", exist_ok=True)
    os.makedirs(os.path.join("data", "contracts"), exist_ok=True)
    os.makedirs("reports", exist_ok=True)
    os.makedirs("artifacts", exist_ok=True)

    # load environment variables
    load_dotenv()

    # setup logger
    log.basicConfig(level=log.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    # obtain and process the data
    df_use = obtain_and_process_data()

    # run the backtest
    backtest(df_use)
