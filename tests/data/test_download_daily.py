from pathlib import Path

import pandas as pd

from mfe5210_alpha.data.download_daily import (
    download_one_security,
    download_universe,
    filter_a_share_universe,
)


class DummyClient:
    def fetch_all_stocks(self, day: str):
        import pandas as pd

        return pd.DataFrame(
            {
                "code": [
                    "sh.000001",
                    "sh.600000",
                    "sh.688001",
                    "sz.000001",
                    "sz.300001",
                    "sh.900901",
                    "sz.200001",
                    "sz.399001",
                ],
                "code_name": ["上证指数", "浦发银行", "华兴源创", "平安银行", "特锐德", "B股", "B股", "深证成指"],
            }
        )

    def fetch_index_stocks(self, index_name: str, day: str):
        import pandas as pd

        assert day == "2026-04-20"
        if index_name == "hs300":
            return pd.DataFrame(
                {
                    "code": ["sh.600000", "sz.000001"],
                    "code_name": ["浦发银行", "平安银行"],
                }
            )
        if index_name == "zz500":
            return pd.DataFrame(
                {
                    "code": ["sz.000001", "sh.600004"],
                    "code_name": ["平安银行", "白云机场"],
                }
            )
        raise AssertionError(index_name)


def test_filter_a_share_universe_removes_indices_and_b_shares():
    universe = DummyClient().fetch_all_stocks("2025-08-29")

    filtered = filter_a_share_universe(universe)

    assert filtered["code"].tolist() == ["sh.600000", "sh.688001", "sz.000001", "sz.300001"]


def test_download_universe_respects_limit(tmp_path: Path):
    called = []

    def _fake_download_one_security(cfg, security, client=None):
        called.append(security)
        path = tmp_path / f"{security}.parquet"
        path.write_text("ok", encoding="utf-8")
        return path

    class DummyCfg:
        raw_root = tmp_path
        sample_start = "2012-01-01"
        sample_end = "2026-04-20"

    paths = download_universe(
        cfg=DummyCfg(),
        trade_date="2026-04-20",
        limit=2,
        client=DummyClient(),
        downloader=_fake_download_one_security,
    )

    assert called == ["sh.600000", "sh.688001"]
    assert len(paths) == 2


def test_download_universe_skips_existing_files(tmp_path: Path):
    existing = tmp_path / "prices" / "sh.600000.parquet"
    existing.parent.mkdir(parents=True)
    pd.DataFrame({"date": ["2026-04-20"], "code": ["sh.600000"]}).to_parquet(existing)
    called = []

    def _fake_download_one_security(cfg, security, client=None):
        called.append(security)
        path = tmp_path / "prices" / f"{security}.parquet"
        path.write_text("downloaded", encoding="utf-8")
        return path

    class DummyCfg:
        raw_root = tmp_path
        sample_start = "2012-01-01"
        sample_end = "2026-04-20"

    paths = download_universe(
        cfg=DummyCfg(),
        trade_date="2026-04-20",
        limit=2,
        client=DummyClient(),
        downloader=_fake_download_one_security,
        skip_existing=True,
    )

    assert called == ["sh.688001"]
    assert paths == [existing, tmp_path / "prices" / "sh.688001.parquet"]


def test_download_universe_can_use_combined_index_universe(tmp_path: Path):
    called = []

    def _fake_download_one_security(cfg, security, client=None):
        called.append(security)
        path = tmp_path / "prices" / f"{security}.parquet"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("ok", encoding="utf-8")
        return path

    class DummyCfg:
        raw_root = tmp_path
        sample_start = "2012-01-01"
        sample_end = "2026-04-20"

    paths = download_universe(
        cfg=DummyCfg(),
        trade_date="2026-04-20",
        universe_name="hs300_zz500",
        client=DummyClient(),
        downloader=_fake_download_one_security,
    )

    assert called == ["sh.600000", "sz.000001", "sh.600004"]
    assert len(paths) == 3


def test_download_one_security_appends_stale_existing_file(tmp_path: Path):
    existing = tmp_path / "prices" / "sh.600000.parquet"
    existing.parent.mkdir(parents=True)
    pd.DataFrame(
        {
            "date": ["2026-04-17"],
            "code": ["sh.600000"],
            "close": ["10.0"],
        }
    ).to_parquet(existing)

    class DummyCfg:
        raw_root = tmp_path
        sample_start = "2012-01-01"
        sample_end = "2026-04-20"

    class PriceClient:
        def __init__(self):
            self.calls = []

        def fetch_price(self, security, start_date, end_date, fields):
            self.calls.append((security, start_date, end_date))
            return pd.DataFrame(
                {
                    "date": ["2026-04-18", "2026-04-20"],
                    "code": [security, security],
                    "close": ["10.1", "10.2"],
                }
            )

    client = PriceClient()

    path = download_one_security(DummyCfg(), "sh.600000", client=client)
    saved = pd.read_parquet(path)

    assert client.calls == [("sh.600000", "2026-04-18", "2026-04-20")]
    assert saved["date"].tolist() == ["2026-04-17", "2026-04-18", "2026-04-20"]
