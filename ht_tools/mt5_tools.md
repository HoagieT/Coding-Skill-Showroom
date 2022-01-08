# Class TickerLinearCombination
## Attributes that need manual input when forming the combination
**`ticker_list`**: The tickers/symbols used to form the linear combination. **No default value**.

**`weights`**: The weights of each ticker. Sequence matches that of the ticker list. **No default value**.

**`name`**: The name of the combination. Name is set to None by default but a real name is strongly recommended.

**`status`**: A combinaiton has one of the five statuses: 1 (long), -1 (short), 0 (not in trading), suspended, terminated. Set to 0 by default.

**`lot`**: The volume of the current position. Lot can be positive or negative, indicating long or short. Set to zero by default.

**`trade_tickets`**: List type. Each element stores the MetaTrader5 ticket of the ticker, set to Nones by default.

**`trade_volumes`**: List type. Each element stores the MetaTrader5 volume of the ticker. In most cases, volume=abs(weight * lot), but occassionally some orders might only be partially filled. In rare cases, volume<abs(weight * lot). Set to zeros by default.

**`margin_occupied`**: List type. Each element stores the margin occupied by the corresponding ticker. Set to zeros by default.

**`open_price`**: Float type, the most recent open price of the combination. Set to None by default.

**`close_price`**: Float type, the most recent close price of the combination. Set to None by default.

**`open_time`**: Float type, the most recent open time of the combination. Set to None by default.

**`close_time`**: Float type, the most recent close time of the combination. Set to None by default.

**`trade_id`**: Int type. Each opened position of the combination will be assigned a random 7-digit id for future reference. Set to None by default.



## Attributes that do not need manual input when forming the combination
**`n_tickers`**: The number of tickers in the combination.

**`contract_size`**: List type. The contract size of each ticker. MetaTrader5 shows the price of each ticker and the tradable contract contains multiple units of the ticker.

**`pnl`**: Float type. If a combination is in trading, this attribute stores the unrealized profit or loss of the combination, i.e. the net sum of p/l on all tickers.

**`filling_type`**: MetaTrader5 has three order filling types, Fill or Kill (FOK), Immediate or Cancel (IOC), and Return. Depending on the broker, each ticker can be traded with one of the first two filling types. This information can ben obtained by calling the MetaTrader5.symbols_get() function.

**`max_volume`**: The maximum allowed lot of the combination.

**`chart`**: Stores either the matplotlib chart of the html chart of the combination.

**`price_history`**: pd.Series type. Index must be datetime type.


## Functions
**`is_active(threshold_sec=60)`**: The number of tickers in the combination. If there has been a quote within *threshold_sec*, e.g. 60 seconds, on all of the tickers, then the combination is considered active.

**`get_pnl()`**: Returns the un realized p/l for a combination in trading, or 0 for a combinaiton not in trading.

**`get_margin_requirement(lot=1)`**: Get required margin for the combination.

