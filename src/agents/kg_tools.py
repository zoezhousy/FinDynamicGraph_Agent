from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List

from src.kg.query import KGQueryClient


class KGAgentContext:
    def __init__(self, query_client: KGQueryClient) -> None:
        self.query_client = query_client

    def load_subgraph(self, ticker: str, trade_date: datetime) -> Dict[str, List[Dict[str, Any]]]:
        return self.query_client.get_ticker_subgraph(ticker, trade_date)

