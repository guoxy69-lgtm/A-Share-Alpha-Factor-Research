import pandas as pd

from mfe5210_alpha.data.baostock_client import BaostockClient


class DummyLoginResult:
    error_code = "0"
    error_msg = "success"


class DummyQueryResult:
    error_code = "0"
    error_msg = "success"
    fields = ["date", "code", "open", "close", "volume"]

    def __init__(self):
        self._rows = [
            ["2020-01-02", "sz.000001", "10.0", "10.2", "100"],
            ["2020-01-03", "sz.000001", "10.5", "10.8", "120"],
        ]
        self._idx = -1

    def next(self):
        self._idx += 1
        return self._idx < len(self._rows)

    def get_row_data(self):
        return self._rows[self._idx]


class DummyStockListResult:
    error_code = "0"
    error_msg = "success"
    fields = ["code", "code_name"]

    def __init__(self):
        self._rows = [
            ["sh.600000", "浦发银行"],
            ["sz.000001", "平安银行"],
        ]
        self._idx = -1

    def next(self):
        self._idx += 1
        return self._idx < len(self._rows)

    def get_row_data(self):
        return self._rows[self._idx]


class DummyBS:
    def login(self, user_id="anonymous", password="123456"):
        assert user_id == "anonymous"
        assert password == "123456"
        return DummyLoginResult()

    def query_history_k_data_plus(self, security, fields, start_date, end_date, frequency, adjustflag):
        assert security == "sz.000001"
        assert "date,code,open,close,volume" == fields
        assert frequency == "d"
        assert adjustflag == "2"
        return DummyQueryResult()

    def query_all_stock(self, day):
        assert day == "2025-08-29"
        return DummyStockListResult()

    def query_hs300_stocks(self, day):
        assert day == "2025-08-29"
        return DummyStockListResult()


def test_baostock_client_downloads_price_frame(monkeypatch):
    monkeypatch.setattr("mfe5210_alpha.data.baostock_client.bs", DummyBS())

    client = BaostockClient()
    df = client.fetch_price(
        security="sz.000001",
        start_date="2020-01-02",
        end_date="2020-01-03",
        fields=["date", "code", "open", "close", "volume"],
    )

    assert list(df.columns) == ["date", "code", "open", "close", "volume"]
    assert df.iloc[0]["code"] == "sz.000001"
    assert len(df) == 2


def test_baostock_client_fetches_stock_universe(monkeypatch):
    monkeypatch.setattr("mfe5210_alpha.data.baostock_client.bs", DummyBS())

    client = BaostockClient()
    df = client.fetch_all_stocks("2025-08-29")

    assert list(df.columns) == ["code", "code_name"]
    assert len(df) == 2
    assert set(df["code"]) == {"sh.600000", "sz.000001"}


def test_baostock_client_fetches_index_constituents(monkeypatch):
    monkeypatch.setattr("mfe5210_alpha.data.baostock_client.bs", DummyBS())

    client = BaostockClient()
    df = client.fetch_index_stocks("hs300", "2025-08-29")

    assert list(df.columns) == ["code", "code_name"]
    assert set(df["code"]) == {"sh.600000", "sz.000001"}
