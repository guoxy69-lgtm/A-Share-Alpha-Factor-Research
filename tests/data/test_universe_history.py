import pandas as pd

from mfe5210_alpha.data.universe_history import (
    build_historical_union,
    build_union_membership,
    expand_membership_snapshots,
)


def test_build_union_membership_keeps_daily_historical_union():
    hs300 = pd.DataFrame(
        {"date": ["2020-01-02", "2020-01-03"], "code": ["sh.600000", "sh.600001"]}
    )
    zz500 = pd.DataFrame(
        {"date": ["2020-01-02", "2020-01-03"], "code": ["sz.000001", "sh.600001"]}
    )

    out = build_union_membership(hs300, zz500)

    assert list(out.columns) == ["date", "security", "member_hs300", "member_zz500"]
    assert set(out.loc[out["date"] == pd.Timestamp("2020-01-02"), "security"]) == {
        "sh.600000",
        "sz.000001",
    }
    assert out.loc[out["security"] == "sh.600001", "member_zz500"].iloc[0] == 1


def test_expand_membership_snapshots_applies_snapshot_until_next_update():
    snapshots = pd.DataFrame(
        {
            "date": pd.to_datetime(["2020-01-01", "2020-01-03"]),
            "security": ["sh.600000", "sz.000001"],
            "member_hs300": [1, 0],
            "member_zz500": [0, 1],
        }
    )
    trade_dates = ["2020-01-02", "2020-01-03", "2020-01-06"]

    out = expand_membership_snapshots(snapshots, trade_dates)

    assert set(out.loc[out["date"] == pd.Timestamp("2020-01-02"), "security"]) == {
        "sh.600000"
    }
    assert set(out.loc[out["date"] == pd.Timestamp("2020-01-03"), "security"]) == {
        "sz.000001"
    }
    assert set(out.loc[out["date"] == pd.Timestamp("2020-01-06"), "security"]) == {
        "sz.000001"
    }


def test_build_historical_union_queries_monthly_snapshots_not_every_day():
    class FakeClient:
        def __init__(self):
            self.index_queries = []

        def fetch_trade_dates(self, start_date, end_date):
            return pd.DataFrame(
                {
                    "date": [
                        "2020-01-02",
                        "2020-01-03",
                        "2020-01-31",
                        "2020-02-03",
                        "2020-02-28",
                    ]
                }
            )

        def fetch_index_stocks(self, index_name, day):
            self.index_queries.append((index_name, day))
            update_date = "2020-01-02" if day < "2020-02-01" else "2020-02-03"
            code = "sh.600000" if index_name == "hs300" else "sz.000001"
            return pd.DataFrame(
                {"updateDate": [update_date], "code": [code], "code_name": [code]}
            )

    client = FakeClient()

    out = build_historical_union(client, "2020-01-01", "2020-02-28")

    assert len(client.index_queries) == 6
    assert ("hs300", "2020-01-03") not in client.index_queries
    assert out["date"].nunique() == 5
    assert set(out.loc[out["date"] == pd.Timestamp("2020-02-28"), "security"]) == {
        "sh.600000",
        "sz.000001",
    }
