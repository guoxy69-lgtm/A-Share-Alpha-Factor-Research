import pandas as pd

from mfe5210_alpha.data.build_research_store import build_research_panel


def test_build_research_panel_creates_returns_and_adv20():
    raw = pd.DataFrame(
        {
            "date": pd.to_datetime(["2020-01-02", "2020-01-03", "2020-01-06"]),
            "code": ["sz.000001"] * 3,
            "open": ["10.0", "10.5", "10.8"],
            "high": ["10.3", "10.8", "11.0"],
            "low": ["9.9", "10.4", "10.7"],
            "close": ["10.2", "10.8", "10.6"],
            "preclose": ["10.0", "10.2", "10.8"],
            "volume": ["100", "120", "90"],
            "amount": ["1000", "1300", "950"],
            "turn": ["0.1", "0.2", "0.15"],
            "tradestatus": ["1", "1", "1"],
            "pctChg": ["2.0", "5.88", "-1.85"],
            "peTTM": ["8.0", "8.2", "8.1"],
            "pbMRQ": ["1.1", "1.2", "1.15"],
            "psTTM": ["3.0", "3.1", "3.05"],
            "pcfNcfTTM": ["-11.0", "-10.5", "-10.8"],
            "isST": ["0", "0", "0"],
        }
    )

    panel = build_research_panel(raw)

    assert "security" in panel.columns
    assert "ret_1d" in panel.columns
    assert "fwd_ret_1d" in panel.columns
    assert "adv20" in panel.columns
    assert round(panel.iloc[1]["ret_1d"], 6) == round(10.8 / 10.2 - 1, 6)


def test_build_research_panel_filters_non_a_share_codes():
    raw = pd.DataFrame(
        {
            "date": pd.to_datetime(["2020-01-02", "2020-01-02"]),
            "code": ["sh.000001", "sh.600000"],
            "open": ["3000.0", "10.0"],
            "high": ["3010.0", "10.3"],
            "low": ["2990.0", "9.9"],
            "close": ["3005.0", "10.2"],
            "preclose": ["3000.0", "10.0"],
            "volume": ["100000", "100"],
            "amount": ["1000000", "1000"],
            "turn": ["", "0.1"],
            "tradestatus": ["1", "1"],
            "pctChg": ["0.1", "2.0"],
            "peTTM": ["", "8.0"],
            "pbMRQ": ["", "1.1"],
            "psTTM": ["", "3.0"],
            "pcfNcfTTM": ["", "-11.0"],
            "isST": ["0", "0"],
        }
    )

    panel = build_research_panel(raw)

    assert panel["security"].tolist() == ["sh.600000"]


def test_build_research_panel_can_filter_allowed_securities():
    raw = pd.DataFrame(
        {
            "date": pd.to_datetime(["2020-01-02", "2020-01-02"]),
            "code": ["sh.600000", "sh.600004"],
            "open": ["10.0", "8.0"],
            "high": ["10.3", "8.2"],
            "low": ["9.9", "7.8"],
            "close": ["10.2", "8.1"],
            "preclose": ["10.0", "8.0"],
            "volume": ["100", "90"],
            "amount": ["1000", "800"],
            "turn": ["0.1", "0.2"],
            "tradestatus": ["1", "1"],
            "pctChg": ["2.0", "1.25"],
            "peTTM": ["8.0", "7.0"],
            "pbMRQ": ["1.1", "1.0"],
            "psTTM": ["3.0", "2.5"],
            "pcfNcfTTM": ["-11.0", "-9.0"],
            "isST": ["0", "0"],
        }
    )

    panel = build_research_panel(raw, allowed_securities={"sh.600000"})

    assert panel["security"].tolist() == ["sh.600000"]
