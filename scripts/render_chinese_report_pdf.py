from __future__ import annotations

import html
import re
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    Image,
    LongTable,
    PageBreak,
    Paragraph,
    Preformatted,
    SimpleDocTemplate,
    Spacer,
    TableStyle,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPORT_ROOT = PROJECT_ROOT / "report_zh"
MAIN_TEX = REPORT_ROOT / "main_zh.tex"
MAIN_PDF = REPORT_ROOT / "main_zh.pdf"
REFERENCES_BIB = PROJECT_ROOT / "report" / "references.bib"
SIMHEI_TTF = Path(r"C:\Windows\Fonts\simhei.ttf")


def _register_font() -> None:
    if not SIMHEI_TTF.exists():
        raise FileNotFoundError(f"Chinese font not found: {SIMHEI_TTF}")
    if "SimHei" not in pdfmetrics.getRegisteredFontNames():
        pdfmetrics.registerFont(TTFont("SimHei", str(SIMHEI_TTF)))


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _clean_inline(text: str) -> str:
    text = re.sub(r"\\citep\{[^}]*\}", "", text)
    text = re.sub(r"\\nocite\{[^}]*\}", "", text)
    text = re.sub(r"\\texttt\{([^}]*)\}", r"\1", text)
    text = re.sub(r"\\emph\{([^}]*)\}", r"\1", text)
    text = re.sub(r"\\ref\{[^}]*\}", "", text)
    text = re.sub(r"\\label\{[^}]*\}", "", text)
    text = re.sub(r"\\[A-Za-z]+\*?\{([^}]*)\}", r"\1", text)
    text = text.replace(r"\%", "%")
    text = text.replace(r"\_", "_")
    text = text.replace(r"\&", "&")
    text = text.replace(r"\$", "$")
    text = text.replace("~", " ")
    text = text.replace("``", '"')
    text = text.replace("''", '"')
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _extract_braced_value(tex: str, command: str) -> str:
    match = re.search(rf"\\{command}\{{(.*?)\}}", tex, flags=re.S)
    return _clean_inline(match.group(1)) if match else ""


def _translate_table_cell(table_name: str, cell: str) -> str:
    common = {
        "Metric": "指标",
        "Value": "数值",
        "Factor": "因子",
        "Weight": "权重",
        "LS Sharpe": "LS 夏普",
        "RankICIR": "RankICIR",
        "Robust Score": "稳健得分",
        "Protocol": "评估协议",
        "Best Model": "最优模型",
        "CAGR": "CAGR",
        "Sharpe": "夏普",
        "Max DD": "最大回撤",
        "Final NAV": "期末净值",
        "Model": "模型",
        "Ann. Return": "年化收益",
        "Turnover": "换手率",
        "Q5-Q1": "Q5-Q1",
        "Trade Window": "交易窗口",
        "Selected Factors": "入选因子",
        "Weights": "权重",
        "Operator": "算子",
        "Definition": "定义",
        "Family": "类别",
        "Fields": "字段",
        "Exact formula": "精确公式",
        "Sign": "方向",
        "F Sharpe": "因子夏普",
        "IC": "IC",
        "RankIC": "RankIC",
        "ICIR": "ICIR",
        "Stage": "筛选阶段",
        "Incr. Sharpe": "增量夏普",
        "Fixed train-test": "固定训练/测试",
        "Rolling walk-forward": "滚动 walk-forward",
        "Fixed Rank-ICIR neutral": "固定 Rank-ICIR 中性模型",
        "Rolling robust-score neutral": "滚动 robust-score 中性模型",
        "Sample start": "样本起点",
        "Sample end": "样本终点",
        "Trading dates": "交易日数",
        "Stock-day observations": "股票-日期观测数",
        "Unique securities": "股票数量",
        "Unique panel securities": "研究面板股票数",
        "Securities with industry label": "有行业标签的股票数",
        "Industry coverage": "行业覆盖率",
        "Baostock industry rows": "Baostock 行业记录数",
        "Distinct industry names": "不同行业名称数",
        "selected": "已入选",
        "correlation_blocked": "相关性拦截",
        "quality_blocked": "质量拦截",
        "increment_blocked": "增量贡献拦截",
        "dropped_unstable_sign": "方向不稳定剔除",
        "not_evaluated": "未评估",
        "momentum": "动量",
        "volatility": "波动率",
        "liquidity": "流动性",
        "valuation": "估值",
        "wq_like": "WQ 风格",
        "delay(x,n): security-level lag of field x by n trading days.": "delay(x,n)：字段 x 在个股维度上滞后 n 个交易日。",
        "mean(x,n): security-level rolling arithmetic mean over n trading days.": "mean(x,n)：字段 x 在个股维度上计算最近 n 个交易日的滚动算术平均。",
        "std(x,n): security-level rolling standard deviation over n trading days.": "std(x,n)：字段 x 在个股维度上计算最近 n 个交易日的滚动标准差。",
        "diff(x,n): security-level x_t minus x_{t-n}.": "diff(x,n)：字段 x 在个股维度上的 x_t 减去 x_{t-n}。",
        "pct_change(x,n): security-level x_t / x_{t-n} - 1.": "pct_change(x,n)：字段 x 在个股维度上的 x_t / x_{t-n} - 1。",
        "zscore(x): cross-sectional daily z-score using same-date mean and population standard deviation.": "zscore(x)：按日期做横截面 z-score，使用同日均值和总体标准差。",
        "winsorize(x): cross-sectional daily clipping at the 1st and 99th percentiles.": "winsorize(x)：按日期在横截面上截尾到 1% 和 99% 分位数。",
        "rankIC": "rankIC",
        "Daily Spearman rank correlation between factor exposure and next-day return.": "因子暴露与下一日收益之间的日度 Spearman 秩相关系数。",
        "Daily Pearson correlation between factor exposure and next-day return.": "因子暴露与下一日收益之间的日度 Pearson 相关系数。",
        "residualize(x,Z)": "residualize(x,Z)",
        "Daily OLS residual from regressing factor x on control matrix Z.": "按日期将因子 x 对控制矩阵 Z 做 OLS 回归后得到的残差。",
    }
    return common.get(cell, cell)


def _parse_table_rows(path: Path) -> list[list[str]]:
    raw = _read_text(path)
    rows: list[list[str]] = []
    for line in raw.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith("\\") and any(
            stripped.startswith(prefix)
            for prefix in (
                r"\scriptsize",
                r"\normalsize",
                r"\setlength",
                r"\begin{tabular}",
                r"\end{tabular}",
                r"\begin{longtable}",
                r"\end{longtable}",
                r"\toprule",
                r"\midrule",
                r"\bottomrule",
                r"\endfirsthead",
                r"\endhead",
            )
        ):
            continue
        if not stripped.endswith(r"\\"):
            continue
        cells = [cell.strip() for cell in stripped[:-2].split("&")]
        cleaned = [
            _translate_table_cell(path.name, _clean_inline(cell).replace("~", " "))
            for cell in cells
        ]
        rows.append(cleaned)
    deduped: list[list[str]] = []
    for row in rows:
        if deduped and row == deduped[-1]:
            continue
        deduped.append(row)
    return deduped


def _parse_references(path: Path) -> list[str]:
    raw = _read_text(path)
    entries = re.findall(r"@\w+\{([^,]+),(.*?)\n\}", raw, flags=re.S)
    rendered = []
    for _, body in entries:
        fields = {
            key.strip(): value.strip()
            for key, value in re.findall(r"(\w+)\s*=\s*\{([^}]*)\}", body, flags=re.S)
        }
        author = fields.get("author", "").strip("{}")
        year = fields.get("year", "")
        title = fields.get("title", "")
        note = fields.get("note", "")
        parts = [part for part in [author, f"({year})" if year else "", title, note] if part]
        rendered.append(" ".join(parts))
    return rendered


def _parse_blocks(tex: str) -> list[tuple[str, object]]:
    body = tex.split(r"\begin{document}", 1)[1].split(r"\end{document}", 1)[0]
    lines = body.splitlines()
    blocks: list[tuple[str, object]] = []
    paragraph_lines: list[str] = []
    current_section = ""
    idx = 0

    def flush_paragraph() -> None:
        nonlocal paragraph_lines
        text = _clean_inline(" ".join(paragraph_lines))
        paragraph_lines = []
        if text:
            blocks.append(("paragraph", text))

    while idx < len(lines):
        stripped = lines[idx].strip()

        if not stripped:
            flush_paragraph()
            idx += 1
            continue

        if stripped in {r"\maketitle", r"\begin{landscape}", r"\end{landscape}"}:
            flush_paragraph()
            idx += 1
            continue

        if stripped == r"\appendix":
            flush_paragraph()
            blocks.append(("pagebreak", None))
            blocks.append(("heading2", "附录"))
            idx += 1
            continue

        if stripped == r"\begin{abstract}":
            flush_paragraph()
            abstract_lines: list[str] = []
            idx += 1
            while idx < len(lines) and lines[idx].strip() != r"\end{abstract}":
                abstract_lines.append(lines[idx].strip())
                idx += 1
            blocks.append(("heading2", "摘要"))
            blocks.append(("paragraph", _clean_inline(" ".join(abstract_lines))))
            idx += 1
            continue

        section_match = re.match(r"\\section\{(.*)\}", stripped)
        if section_match:
            flush_paragraph()
            current_section = _clean_inline(section_match.group(1))
            blocks.append(("heading2", current_section))
            idx += 1
            continue

        subsection_match = re.match(r"\\subsection\{(.*)\}", stripped)
        if subsection_match:
            flush_paragraph()
            blocks.append(("heading3", _clean_inline(subsection_match.group(1))))
            idx += 1
            continue

        if stripped in {r"\begin{itemize}", r"\begin{enumerate}"}:
            flush_paragraph()
            ordered = stripped == r"\begin{enumerate}"
            items: list[str] = []
            idx += 1
            while idx < len(lines):
                current = lines[idx].strip()
                if current in {r"\end{itemize}", r"\end{enumerate}"}:
                    break
                if current.startswith(r"\item"):
                    items.append(_clean_inline(current[len(r"\item") :].strip()))
                idx += 1
            blocks.append(("list", (ordered, items)))
            idx += 1
            continue

        if stripped == r"\[":
            flush_paragraph()
            eq_lines: list[str] = []
            idx += 1
            while idx < len(lines) and lines[idx].strip() != r"\]":
                eq_lines.append(lines[idx].rstrip())
                idx += 1
            blocks.append(("equation", "\n".join(eq_lines).strip()))
            idx += 1
            continue

        if stripped == r"\begin{table}[H]":
            flush_paragraph()
            caption = ""
            input_path: Path | None = None
            idx += 1
            while idx < len(lines) and lines[idx].strip() != r"\end{table}":
                current = lines[idx].strip()
                caption_match = re.match(r"\\caption\{(.*)\}", current)
                if caption_match:
                    caption = _clean_inline(caption_match.group(1))
                input_match = re.search(r"input\{([^}]+)\}", current)
                if input_match:
                    input_path = (REPORT_ROOT / input_match.group(1)).resolve()
                idx += 1
            if input_path is not None:
                blocks.append(("table", (caption or current_section, input_path)))
            idx += 1
            continue

        if stripped == r"\begin{figure}[H]":
            flush_paragraph()
            caption = ""
            image_path: Path | None = None
            idx += 1
            while idx < len(lines) and lines[idx].strip() != r"\end{figure}":
                current = lines[idx].strip()
                image_match = re.search(r"includegraphics(?:\[[^\]]*\])?\{([^}]+)\}", current)
                if image_match:
                    image_path = (REPORT_ROOT / image_match.group(1)).resolve()
                caption_match = re.match(r"\\caption\{(.*)\}", current)
                if caption_match:
                    caption = _clean_inline(caption_match.group(1))
                idx += 1
            if image_path is not None:
                blocks.append(("figure", (caption or image_path.name, image_path)))
            idx += 1
            continue

        input_match = re.match(r"\\input\{([^}]+)\}", stripped)
        if input_match:
            flush_paragraph()
            blocks.append(("table", (current_section, (REPORT_ROOT / input_match.group(1)).resolve())))
            idx += 1
            continue

        if stripped.startswith(r"\bibliographystyle") or stripped.startswith(r"\bibliography") or stripped.startswith(r"\nocite"):
            flush_paragraph()
            idx += 1
            continue

        if stripped.startswith("%"):
            idx += 1
            continue

        paragraph_lines.append(stripped)
        idx += 1

    flush_paragraph()
    return blocks


def _styles():
    _register_font()
    styles = getSampleStyleSheet()
    styles.add(
        ParagraphStyle(
            name="TitleCenter",
            parent=styles["Title"],
            alignment=TA_CENTER,
            fontName="SimHei",
            fontSize=18.5,
            leading=24,
            spaceAfter=8,
            wordWrap="CJK",
        )
    )
    styles.add(
        ParagraphStyle(
            name="MetaCenter",
            parent=styles["Normal"],
            alignment=TA_CENTER,
            fontName="SimHei",
            fontSize=10.8,
            leading=14,
            spaceAfter=2,
            wordWrap="CJK",
        )
    )
    styles.add(
        ParagraphStyle(
            name="Body",
            parent=styles["BodyText"],
            fontName="SimHei",
            fontSize=10.2,
            leading=14,
            spaceAfter=8,
            wordWrap="CJK",
        )
    )
    styles.add(
        ParagraphStyle(
            name="Heading2Custom",
            parent=styles["Heading2"],
            fontName="SimHei",
            fontSize=14.5,
            leading=18,
            spaceBefore=12,
            spaceAfter=8,
            wordWrap="CJK",
        )
    )
    styles.add(
        ParagraphStyle(
            name="Heading3Custom",
            parent=styles["Heading3"],
            fontName="SimHei",
            fontSize=11.5,
            leading=15,
            spaceBefore=10,
            spaceAfter=6,
            wordWrap="CJK",
        )
    )
    styles.add(
        ParagraphStyle(
            name="Caption",
            parent=styles["Normal"],
            fontName="SimHei",
            fontSize=9.4,
            leading=11.5,
            alignment=TA_CENTER,
            spaceAfter=6,
            wordWrap="CJK",
        )
    )
    styles.add(
        ParagraphStyle(
            name="TableCell",
            parent=styles["Normal"],
            fontName="SimHei",
            fontSize=7.2,
            leading=8.9,
            wordWrap="CJK",
        )
    )
    styles.add(
        ParagraphStyle(
            name="TableCellTiny",
            parent=styles["Normal"],
            fontName="SimHei",
            fontSize=6.5,
            leading=8.0,
            wordWrap="CJK",
        )
    )
    styles.add(
        ParagraphStyle(
            name="Ref",
            parent=styles["Normal"],
            fontName="SimHei",
            fontSize=9.8,
            leading=13.2,
            leftIndent=14,
            firstLineIndent=-14,
            spaceAfter=6,
            wordWrap="CJK",
        )
    )
    return styles


def _table_col_widths(name: str, doc_width: float) -> list[float] | None:
    if name in {"research_panel_summary.tex", "industry_coverage.tex"}:
        return [doc_width * 0.62, doc_width * 0.38]
    if name == "fixed_best_weights.tex":
        return [doc_width * 0.34, doc_width * 0.12, doc_width * 0.16, doc_width * 0.18, doc_width * 0.20]
    if name in {"oos_fixed_comparison.tex", "oos_rolling_comparison.tex"}:
        return [doc_width * 0.44, doc_width * 0.14, doc_width * 0.14, doc_width * 0.14, doc_width * 0.14]
    if name == "oos_best_summary.tex":
        return [doc_width * 0.20, doc_width * 0.33, doc_width * 0.12, doc_width * 0.11, doc_width * 0.12, doc_width * 0.12]
    if name == "quantile_diagnostics.tex":
        return [doc_width * 0.28] + [doc_width * 0.12] * 6
    if name == "operator_definitions.tex":
        return [doc_width * 0.28, doc_width * 0.72]
    if name == "factor_dictionary.tex":
        return [doc_width * 0.18, doc_width * 0.13, doc_width * 0.19, doc_width * 0.50]
    if name == "single_factor_full_results.tex":
        return [
            doc_width * 0.19,
            doc_width * 0.05,
            doc_width * 0.08,
            doc_width * 0.08,
            doc_width * 0.07,
            doc_width * 0.07,
            doc_width * 0.08,
            doc_width * 0.08,
            doc_width * 0.15,
            doc_width * 0.15,
        ]
    if name == "selection_details.tex":
        return [doc_width * 0.20, doc_width * 0.40, doc_width * 0.40]
    return None


def _build_table(caption: str, path: Path, styles, doc_width: float):
    rows = _parse_table_rows(path)
    if not rows:
        return [Paragraph(caption, styles["Caption"])]
    style_name = "TableCellTiny" if len(rows[0]) >= 8 else "TableCell"
    cell_style = styles[style_name]
    formatted = [
        [Paragraph(html.escape(cell), cell_style) for cell in row]
        for row in rows
    ]
    table = LongTable(
        formatted,
        repeatRows=1,
        colWidths=_table_col_widths(path.name, doc_width),
        splitByRow=1,
    )
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#EDEDED")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.black),
                ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#BDBDBD")),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LEFTPADDING", (0, 0), (-1, -1), 4),
                ("RIGHTPADDING", (0, 0), (-1, -1), 4),
                ("TOPPADDING", (0, 0), (-1, -1), 3),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
            ]
        )
    )
    return [Paragraph(caption, styles["Caption"]), table, Spacer(1, 0.16 * inch)]


def _scaled_image(path: Path, max_width: float):
    image = Image(str(path))
    if image.drawWidth > max_width:
        ratio = max_width / image.drawWidth
        image.drawWidth *= ratio
        image.drawHeight *= ratio
    return image


def _add_page_number(canvas, doc) -> None:
    canvas.saveState()
    canvas.setFont("SimHei", 9)
    canvas.setFillColor(colors.HexColor("#666666"))
    canvas.drawRightString(doc.pagesize[0] - doc.rightMargin, 0.45 * inch, f"第 {doc.page} 页")
    canvas.restoreState()


def render_pdf() -> None:
    tex = _read_text(MAIN_TEX)
    title = _extract_braced_value(tex, "title")
    author = _extract_braced_value(tex, "author")
    date = _extract_braced_value(tex, "date")
    blocks = _parse_blocks(tex)
    refs = _parse_references(REFERENCES_BIB)

    doc = SimpleDocTemplate(
        str(MAIN_PDF),
        pagesize=A4,
        leftMargin=0.65 * inch,
        rightMargin=0.65 * inch,
        topMargin=0.7 * inch,
        bottomMargin=0.6 * inch,
        title=title,
        author=author,
    )
    styles = _styles()
    story = [
        Paragraph(html.escape(title), styles["TitleCenter"]),
        Paragraph(html.escape(author), styles["MetaCenter"]),
        Paragraph(html.escape(date), styles["MetaCenter"]),
        Spacer(1, 0.18 * inch),
    ]

    for block_type, payload in blocks:
        if block_type == "pagebreak":
            story.append(PageBreak())
            continue
        if block_type == "heading2":
            story.append(Paragraph(html.escape(str(payload)), styles["Heading2Custom"]))
            continue
        if block_type == "heading3":
            story.append(Paragraph(html.escape(str(payload)), styles["Heading3Custom"]))
            continue
        if block_type == "paragraph":
            story.append(Paragraph(html.escape(str(payload)), styles["Body"]))
            continue
        if block_type == "list":
            ordered, items = payload
            for idx, item in enumerate(items, start=1):
                prefix = f"{idx}. " if ordered else u"\u2022 "
                story.append(Paragraph(html.escape(prefix + item), styles["Body"]))
            continue
        if block_type == "equation":
            story.append(
                Preformatted(
                    str(payload),
                    ParagraphStyle(
                        "EquationPreZH",
                        fontName="Courier",
                        fontSize=8.2,
                        leading=10.0,
                        backColor=colors.HexColor("#F7F7F7"),
                        borderPadding=6,
                        borderColor=colors.HexColor("#D6D6D6"),
                        borderWidth=0.5,
                        spaceBefore=4,
                        spaceAfter=8,
                    ),
                )
            )
            continue
        if block_type == "table":
            caption, table_path = payload
            story.extend(_build_table(str(caption), Path(table_path), styles, doc.width))
            continue
        if block_type == "figure":
            caption, image_path = payload
            story.append(_scaled_image(Path(image_path), doc.width))
            story.append(Paragraph(html.escape(str(caption)), styles["Caption"]))
            story.append(Spacer(1, 0.14 * inch))
            continue

    story.append(Paragraph("参考文献", styles["Heading2Custom"]))
    for idx, ref in enumerate(refs, start=1):
        story.append(Paragraph(html.escape(f"[{idx}] {ref}"), styles["Ref"]))

    doc.build(story, onFirstPage=_add_page_number, onLaterPages=_add_page_number)


if __name__ == "__main__":
    render_pdf()
