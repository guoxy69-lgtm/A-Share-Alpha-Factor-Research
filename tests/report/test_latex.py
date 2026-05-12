from pathlib import Path

from mfe5210_alpha.report.latex import build_latex_commands


def test_build_latex_commands_prefers_tectonic():
    commands = build_latex_commands(
        Path(r"E:\MFE5210算法交易\assignment\MFE5210_AlphaFactors_Guoxy69\report"),
        "main.tex",
        engine="tectonic",
    )

    assert commands == [["tectonic", "main.tex"]]


def test_build_latex_commands_treats_tectonic_executable_path_like_tectonic():
    engine = str(Path("tools") / "tectonic.exe")

    commands = build_latex_commands(Path("report"), "main.tex", engine=engine)

    assert commands == [[engine, "main.tex"]]
