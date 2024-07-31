import logging
from decimal import Decimal
from typing import Dict, List, Optional, Set, Tuple

from pydantic import Field

from hummingbot.client.config.config_data_types import ClientFieldData
from hummingbot.core.data_type.common import OrderType, TradeType
from hummingbot.core.data_type.order_book import OrderBook
from hummingbot.core.rate_oracle.rate_oracle import RateOracle
from hummingbot.data_feed.candles_feed.data_types import CandlesConfig
from hummingbot.strategy_v2.controllers import ControllerBase, ControllerConfigBase
from hummingbot.strategy_v2.executors.position_executor.data_types import PositionExecutorConfig, TripleBarrierConfig
from hummingbot.strategy_v2.models.executor_actions import CreateExecutorAction, ExecutorAction, StopExecutorAction
from hummingbot.strategy_v2.models.executors import CloseType


class PMMFairPriceConfig(ControllerConfigBase):
    controller_name = "pmm_fair_price"
    # As this controller is a simple version of the PMM, we are not using the candles feed
    candles_config: List[CandlesConfig] = Field(default=[], client_data=ClientFieldData(prompt_on_new=False))

    taker_connector_name: str = Field(
        default="xrpl",
        client_data=ClientFieldData(
            prompt_on_new=True,
            prompt=lambda mi: "Enter the name of the taker exchange to trade on (e.g., xrpl):"))
    taker_trading_pair: str = Field(
        default="SOLO-XRP",
        client_data=ClientFieldData(
            prompt_on_new=True,
            prompt=lambda mi: "Enter the taker trading pair to trade on (e.g., SOLO-XRP):"))
    taker_conversion_pair: Optional[str] = Field(
        default="XRP-USDT",
        client_data=ClientFieldData(
            prompt_on_new=True,
            prompt=lambda mi: "Enter the taker conversion pair (e.g., XRP-USDT):"))
    maker_connector_name: str = Field(
        default="mexc",
        client_data=ClientFieldData(
            prompt_on_new=True,
            prompt=lambda mi: "Enter the name of the maker exchange to trade on (e.g., mexc):"))
    maker_trading_pair: str = Field(
        default="SOLO-USDT",
        client_data=ClientFieldData(
            prompt_on_new=True,
            prompt=lambda mi: "Enter the maker trading pair to trade on (e.g., SOLO-USDT):"))

    taker_base_amount: float = Field(
        default=25000,
        client_data=ClientFieldData(
            is_updatable=True,
            prompt_on_new=True,
            prompt=lambda mi: "Enter the total base amount for VWAP calculation for taker instrument: "))

    maker_base_amount: float = Field(
        default=25000,
        client_data=ClientFieldData(
            is_updatable=True,
            prompt_on_new=True,
            prompt=lambda mi: "Enter the total base amount for VWAP calculation for maker instrument: "))
    place_live_orders: bool = Field(
        default=False,
        client_data=ClientFieldData(
            is_updatable=True,
            prompt_on_new=True,
            prompt=lambda mi: "Enable/Disable Live Order Placement: "))
    executor_refresh_time: int = Field(
        default=60 * 5,
        client_data=ClientFieldData(
            is_updatable=True,
            prompt_on_new=True,
            prompt=lambda mi: "Enter the refresh time in seconds for executors (e.g., 300 for 5 minutes):"))
    cooldown_time: int = Field(
        default=15,
        client_data=ClientFieldData(
            is_updatable=True,
            prompt_on_new=False,
            prompt=lambda
            mi: "Specify the cooldown time in seconds between after replacing an executor that traded (e.g., 15):"))
    quote_base_amount: float = Field(
        default=0.1,
        client_data=ClientFieldData(
            is_updatable=True,
            prompt_on_new=True,
            prompt=lambda mi: "Total base amount to quote on both sides: "))

    quote_min_tick: Decimal = Field(
        default=0.00001,
        client_data=ClientFieldData(
            is_updatable=True,
            prompt=lambda e: "Enter the min tick size for the quote asset: ",
            prompt_on_new=True
        ))

    pct_spread_max_taker_price_improvement: Decimal = Field(
        default=0.25,
        client_data=ClientFieldData(
            is_updatable=True,
            prompt=lambda e: "Percentage of the taker spread permitted as max price improvement: ",
            prompt_on_new=True
        ))

    @property
    def triple_barrier_config(self) -> TripleBarrierConfig:
        return TripleBarrierConfig(
            stop_loss=None,
            take_profit=None,
            time_limit=None,
            trailing_stop=None,
            open_order_type=OrderType.LIMIT,  # Defaulting to LIMIT as is a Maker Controller
            take_profit_order_type=OrderType.MARKET,
            stop_loss_order_type=OrderType.MARKET,  # Defaulting to MARKET as per requirement
            time_limit_order_type=OrderType.MARKET  # Defaulting to MARKET as per requirement
        )

    def update_markets(self, markets: Dict[str, Set[str]]) -> Dict[str, Set[str]]:
        if self.maker_connector_name not in markets:
            markets[self.maker_connector_name] = set()
        markets[self.maker_connector_name].add(self.maker_trading_pair)
        if self.taker_connector_name not in markets:
            markets[self.taker_connector_name] = set()
        markets[self.taker_connector_name].add(self.taker_trading_pair)
        return markets


class PMMFairPriceController(ControllerBase):

    def __init__(self, config: PMMFairPriceConfig, *args, **kwargs):
        super().__init__(config, *args, **kwargs)
        self.config = config
        self.taker_info = ""
        self.maker_info = ""
        self.info_1 = ""
        self.info_2 = ""
        self.quote_bid = Decimal(0)
        self.quote_ask = Decimal(0)

    def determine_executor_actions(self) -> List[ExecutorAction]:
        """
        Determine actions based on the provided executor handler report.
        """
        actions = []
        actions.extend(self.create_actions_proposal())
        actions.extend(self.stop_actions_proposal())
        return actions

    def create_actions_proposal(self) -> List[ExecutorAction]:
        """
        Create actions proposal based on the current state of the controller.
        """
        create_actions = []
        levels_to_execute = self.get_levels_to_execute()
        for level_id in levels_to_execute:
            price, amount = self.get_price_and_amount(level_id)
            executor_config = self.get_executor_config(level_id, price, amount)
            if executor_config is not None:
                create_actions.append(CreateExecutorAction(
                    controller_id=self.config.id,
                    executor_config=executor_config
                ))
        return create_actions

    def stop_actions_proposal(self) -> List[ExecutorAction]:
        """
        Create a list of actions to stop the executors based on order refresh and early stop conditions.
        """
        stop_actions = []
        stop_actions.extend(self.executors_to_refresh())
        stop_actions.extend(self.executors_to_early_stop())
        return stop_actions

    async def update_processed_data(self):
        # conversion_rate = Decimal("0.5965")
        conversion_rate = Decimal("1")
        if self.config.taker_conversion_pair is not None:
            rate = RateOracle.get_instance().get_pair_rate(self.config.taker_conversion_pair)
            if rate is None:
                logging.getLogger().warning(f"rate for {self.config.taker_conversion_pair} is not ready")
                return
            conversion_rate = Decimal(str(rate))
        # print(conversion_rate)

        taker_order_book: OrderBook = self.market_data_provider.get_order_book(self.config.taker_connector_name,
                                                                               self.config.taker_trading_pair)

        taker_bid, taker_bid_vol = self.get_taker_best_price_and_amount_excluding_own(taker_order_book, TradeType.BUY)
        taker_ask, taker_ask_vol = self.get_taker_best_price_and_amount_excluding_own(taker_order_book, TradeType.SELL)

        taker_mid = (taker_bid + taker_ask) / float(2.0)
        taker_spread = taker_ask - taker_bid
        taker_spread_pct = taker_spread / taker_mid

        self.taker_info = (f"\n== {self.config.taker_connector_name} {self.config.taker_trading_pair} =="
                           f"\nask {taker_ask:.8f} {taker_ask_vol}"
                           f"\nbid {taker_bid:.8f} {taker_bid_vol}"
                           f"\nspread {taker_spread:.8f} {taker_spread_pct:%}")

        taker_vwap_ask = Decimal(str(self.market_data_provider.get_price_for_volume(self.config.taker_connector_name,
                                                                                    self.config.taker_trading_pair,
                                                                                    self.config.taker_base_amount,
                                                                                    True).result_price))

        taker_vwap_bid = Decimal(str(self.market_data_provider.get_price_for_volume(self.config.taker_connector_name,
                                                                                    self.config.taker_trading_pair,
                                                                                    self.config.taker_base_amount,
                                                                                    False).result_price))

        taker_vwap_mid = (taker_vwap_bid + taker_vwap_ask) / Decimal("2.0")

        maker_order_book = self.market_data_provider.get_order_book(self.config.maker_connector_name,
                                                                    self.config.maker_trading_pair)

        maker_bid = list(maker_order_book.bid_entries())[:1][0].price
        maker_bid_vol = list(maker_order_book.bid_entries())[:1][0].amount

        maker_ask = list(maker_order_book.ask_entries())[:1][0].price
        maker_ask_vol = list(maker_order_book.ask_entries())[:1][0].amount

        maker_mid = (maker_bid + maker_ask) / float(2.0)
        maker_spread = maker_ask - maker_mid
        maker_spread_pct = maker_spread / maker_mid

        self.maker_info = (f"\n== {self.config.maker_connector_name} {self.config.maker_trading_pair} =="
                           f"\nask {maker_ask:.8f} ({(Decimal(maker_ask) / conversion_rate):.8f}) {maker_ask_vol:.8f}"
                           f"\nbid {maker_bid:.8f} ({(Decimal(maker_mid) / conversion_rate):.8f}) {maker_bid_vol:.8f}"
                           f"\nspread {maker_spread:.8f} {maker_spread_pct:%}"
                           f"\n{self.config.taker_conversion_pair} conversion rate {conversion_rate:.8f}")

        maker_vwap_ask = Decimal(
            str(self.market_data_provider.get_price_for_volume(self.config.maker_connector_name,
                                                               self.config.maker_trading_pair,
                                                               self.config.maker_base_amount, True).result_price))

        maker_vwap_bid = Decimal(
            str(self.market_data_provider.get_price_for_volume(self.config.maker_connector_name,
                                                               self.config.maker_trading_pair,
                                                               self.config.maker_base_amount, False).result_price))

        maker_vwap_mid = (maker_vwap_bid + maker_vwap_ask) / Decimal("2.0")
        maker_vwap_mid = maker_vwap_mid / conversion_rate

        diff = taker_vwap_mid - maker_vwap_mid
        pct_diff = (diff / maker_vwap_bid)
        q_bid = maker_vwap_mid * (Decimal("1.0") - (Decimal(maker_spread_pct) / Decimal("2.0")))
        q_ask = maker_vwap_mid * (Decimal("1.0") + (Decimal(maker_spread_pct) / Decimal("2.0")))

        self.info_1 = (f"\n== vwap =="
                       f"\ntaker vwap mid {taker_vwap_mid:.8f}"
                       f"\nmaker vwap mid {maker_vwap_mid:.8f}"
                       f"\ndiff {diff:.8f} {pct_diff:.8%}")

        # TODO alexm ideally this price quantizing would be done by market connector
        q_bid = self.quantize_price(q_bid)
        q_ask = self.quantize_price(q_ask)

        max_taker_price_improvement = Decimal(str(taker_spread)) * self.config.pct_spread_max_taker_price_improvement
        max_taker_bid = Decimal(str(taker_bid)) + max_taker_price_improvement
        max_taker_ask = Decimal(str(taker_ask)) - max_taker_price_improvement
        max_taker_bid = self.quantize_price(max_taker_bid)
        max_taker_ask = self.quantize_price(max_taker_ask)

        if q_bid > max_taker_bid:
            self.quote_bid = Decimal(str(max_taker_bid))
            new_mid = self.quote_bid / (Decimal("1.0") - (Decimal(maker_spread_pct) / Decimal("2.0")))
            self.quote_ask = new_mid * (Decimal("1.0") + (Decimal(maker_spread_pct) / Decimal("2.0")))
            self.quote_ask = self.quantize_price(self.quote_ask)
            self.info_2 = (f"\n== quoting join best bid =="
                           f"\ntaker ask {taker_ask:.8f}"
                           f"\nquote ask [{self.quote_ask:.8f} ({self.quote_ask_diff()})]"
                           f"\n--"
                           f"\nquote bid [{self.quote_bid:.8f} ({self.quote_bid_diff()})] < computed {q_bid:.8f}")
        elif q_ask < max_taker_ask:
            self.quote_ask = Decimal(str(max_taker_ask))
            new_mid = self.quote_ask / (Decimal("1.0") + (Decimal(maker_spread_pct) / Decimal("2.0")))
            self.quote_bid = new_mid * (Decimal("1.0") - (Decimal(maker_spread_pct) / Decimal("2.0")))
            self.quote_bid = self.quantize_price(self.quote_bid)

            self.info_2 = (f"\n== quoting join best ask =="
                           f"\nquote ask [{self.quote_ask:.8f} ({self.quote_ask_diff()})] > computed {q_ask:.8f}"
                           f"\n--"
                           f"\nquote bid [{self.quote_bid:.8f}] ({self.quote_bid_diff()})"
                           f"\ntaker bid {taker_bid:.8f}")
        else:
            self.quote_bid = q_bid
            self.quote_ask = q_ask
            self.info_2 = (f"\n== quoting within spread =="
                           f"\ntaker ask [{taker_ask:.8f}]"
                           f"\nquote ask [{self.quote_ask:.8f} ({self.quote_ask_diff()})]"
                           f"\n--"
                           f"\nquote bid [{self.quote_bid:.8f} ({self.quote_bid_diff()})]"
                           f"\ntaker bid {taker_bid:.8f}")

        # logging.getLogger().info(f"quote bid {self.quote_bid} ask {self.quote_ask}")

        self.processed_data = {"quote_bid": self.quote_bid, "quote_ask": self.quote_ask}

    def quote_bid_diff(self) -> str:
        our_bid, our_bid_amount = self.get_current_order_price_and_amount(TradeType.BUY)
        if our_bid is None or self.quote_bid.is_zero():
            return "no bid"
        else:
            return f"our bid {our_bid} {((our_bid - self.quote_bid) / self.quote_bid):.3%}"

    def quote_ask_diff(self) -> str:
        our_ask, our_ask_amount = self.get_current_order_price_and_amount(TradeType.SELL)
        if our_ask is None or self.quote_ask.is_zero():
            return "no ask"
        else:
            return f"our ask {our_ask} {((our_ask - self.quote_ask) / self.quote_ask):.3%}"

    def quantize_price(self, price: Decimal) -> Decimal:
        # return (price // self.config.quote_min_tick) * self.config.quote_min_tick
        return price.quantize(Decimal(self.config.quote_min_tick))

    def get_taker_best_price_and_amount_excluding_own(self, order_book: OrderBook, trade_type: TradeType) -> tuple[
            float, float]:
        our_price, our_amount = self.get_current_order_price_and_amount(trade_type)
        if trade_type == TradeType.BUY:
            top_orders = list(order_book.bid_entries())[:2]
        else:
            top_orders = list(order_book.ask_entries())[:2]
        # if self.quantize_price(Decimal(str(top_orders[0].price))) == our_price and Decimal(str(top_orders[0].amount)) <= our_amount:
        if our_price and our_amount and Decimal(str(top_orders[0].amount)) <= our_amount:
            # logging.getLogger().info(f"excluding top of book as it's our own {trade_type} order")
            return top_orders[1].price, top_orders[1].amount
        else:
            return top_orders[0].price, top_orders[0].amount

    def get_current_order_price_and_amount(self, trade_type: TradeType) -> tuple[Decimal | None, Decimal] | tuple[
            None, None]:
        active_executor = self.filter_executors(
            executors=self.executors_info,
            filter_func=lambda x: x.is_active and self.get_trade_type_from_level_id(
                x.custom_info["level_id"]) == trade_type
        )
        if len(active_executor) > 0:
            return active_executor[0].config.entry_price, active_executor[0].config.amount
        else:
            return None, None

    def get_price_and_amount(self, level_id: str) -> Tuple[Decimal, Decimal]:
        """
        Get the spread and amount in quote for a given level id.
        """
        trade_type = self.get_trade_type_from_level_id(level_id)
        order_price = self.processed_data["quote_bid"] if trade_type == TradeType.BUY else self.processed_data[
            "quote_ask"]
        return order_price, Decimal(str(self.config.quote_base_amount))

    def get_levels_to_execute(self) -> List[str]:
        working_levels = self.filter_executors(
            executors=self.executors_info,
            filter_func=lambda x: x.is_active or (
                x.close_type == CloseType.STOP_LOSS and self.market_data_provider.time()
                - x.close_timestamp < self.config.cooldown_time)
        )
        working_levels_ids = [executor.custom_info["level_id"] for executor in working_levels]
        return self.get_not_active_levels_ids(working_levels_ids)

    def get_level_id_from_side(self, trade_type: TradeType, level: int) -> str:
        """
        Get the level id based on the trade type and the level.
        """
        return f"{trade_type.name.lower()}_{level}"

    def get_trade_type_from_level_id(self, level_id: str) -> TradeType:
        return TradeType.BUY if level_id.startswith("buy") else TradeType.SELL

    def get_level_from_level_id(self, level_id: str) -> int:
        return int(level_id.split('_')[1])

    def get_not_active_levels_ids(self, active_levels_ids: List[str]) -> List[str]:
        """
        Get the levels to execute based on the current state of the controller.
        """

        if "quote_bid" not in self.processed_data or "quote_ask" not in self.processed_data:
            return []

        if not self.config.place_live_orders:
            return []
        buy_ids_missing = [self.get_level_id_from_side(TradeType.BUY, level) for level in range(1)
                           if self.get_level_id_from_side(TradeType.BUY, level) not in active_levels_ids]
        sell_ids_missing = [self.get_level_id_from_side(TradeType.SELL, level) for level in range(1)
                            if self.get_level_id_from_side(TradeType.SELL, level) not in active_levels_ids]
        return buy_ids_missing + sell_ids_missing

    def executors_to_early_stop(self) -> List[ExecutorAction]:
        """
        Get the executors to early stop based on the current state of market data. This method can be overridden to
        implement custom behavior.
        """
        executors_to_refresh = self.filter_executors(
            executors=self.executors_info,
            filter_func=lambda x: not self.config.place_live_orders)

        return [StopExecutorAction(
            controller_id=self.config.id,
            executor_id=executor.id) for executor in executors_to_refresh]

    def executors_to_refresh(self) -> List[ExecutorAction]:
        executors_to_refresh = self.filter_executors(
            executors=self.executors_info,
            filter_func=lambda
            x: not x.is_trading and x.is_active and self.market_data_provider.time() - x.timestamp > self.config.executor_refresh_time)

        if len(executors_to_refresh) > 0:
            # if any executor needs refreshing - refresh all the active executors
            logging.getLogger().info(
                f"will refresh all executors as {self.config.executor_refresh_time} seconds have elapsed")
            executors_to_refresh = self.filter_executors(
                executors=self.executors_info,
                filter_func=lambda x: x.is_active)

        return [StopExecutorAction(
            controller_id=self.config.id,
            executor_id=executor.id) for executor in executors_to_refresh]

    def get_executor_config(self, level_id: str, price: Decimal, amount: Decimal):
        trade_type = self.get_trade_type_from_level_id(level_id)
        return PositionExecutorConfig(
            timestamp=self.market_data_provider.time(),
            level_id=level_id,
            connector_name=self.config.taker_connector_name,
            trading_pair=self.config.taker_trading_pair,
            entry_price=price,
            amount=amount,
            triple_barrier_config=self.config.triple_barrier_config,
            leverage=1,
            side=trade_type,
        )

    def to_format_status(self) -> list[str]:
        lines = []
        lines.extend([self.taker_info, self.maker_info, self.info_1, self.info_2])
        # for executor in self.executors_info:
        #     lines.extend([str(executor)])
        return lines
