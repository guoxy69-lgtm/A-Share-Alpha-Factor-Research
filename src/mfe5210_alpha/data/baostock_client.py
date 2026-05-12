from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

try:
    import baostock as bs  # type: ignore
except ImportError:  # pragma: no cover - exercised via monkeypatch in tests
    class _MissingBaoStockSDK:
        def __getattr__(self, name: str):
            raise RuntimeError("baostock is required for runtime data access")

    bs = _MissingBaoStockSDK()


@dataclass
class BaostockClient:
    user_id: str = "anonymous"
    password: str = "123456"

    def __post_init__(self) -> None:
        login_result = bs.login(user_id=self.user_id, password=self.password)
        if login_result.error_code != "0":
            raise RuntimeError(
                f"baostock login failed: {login_result.error_code} {login_result.error_msg}"
            )

    def _query_to_frame(self, query_result, context: str) -> pd.DataFrame:
        if query_result.error_code != "0":
            raise RuntimeError(
                f"baostock {context} query failed: {query_result.error_code} {query_result.error_msg}"
            )

        rows = []
        while query_result.next():
            rows.append(query_result.get_row_data())
        return pd.DataFrame(rows, columns=query_result.fields)

    def fetch_price(
        self,
        security: str,
        start_date: str,
        end_date: str,
        fields: list[str],
        adjustflag: str = "2",
    ) -> pd.DataFrame:
        rs = bs.query_history_k_data_plus(
            security,
            ",".join(fields),
            start_date=start_date,
            end_date=end_date,
            frequency="d",
            adjustflag=adjustflag,
        )
        return self._query_to_frame(rs, f"price for {security}")

    def fetch_all_stocks(self, day: str) -> pd.DataFrame:
        rs = bs.query_all_stock(day)
        return self._query_to_frame(rs, "stock-universe")

    def fetch_index_stocks(self, index_name: str, day: str) -> pd.DataFrame:
        query_name_map = {
            "sz50": "query_sz50_stocks",
            "hs300": "query_hs300_stocks",
            "zz500": "query_zz500_stocks",
        }
        if index_name not in query_name_map:
            raise ValueError(f"unsupported index universe: {index_name}")
        rs = getattr(bs, query_name_map[index_name])(day)
        return self._query_to_frame(rs, f"{index_name}-universe")
