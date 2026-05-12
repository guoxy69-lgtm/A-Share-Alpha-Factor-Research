from __future__ import annotations

import html
import re
import subprocess
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPORT_ROOT = PROJECT_ROOT / "report"
MAIN_TEX = REPORT_ROOT / "main.tex"
MAIN_HTML = REPORT_ROOT / "main.html"
MAIN_PDF = REPORT_ROOT / "main.pdf"
REFERENCES_BIB = REPORT_ROOT / "references.bib"


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
    text = text.replace("``", '"')
    text = text.replace("''", '"')
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _equation_to_html(lines: list[str]) -> str:
    body = "\n".join(line.rstrip() for line in lines).strip()
    if not body:
        return ""
    safe = html.escape(body.replace(r"\%", "%").replace(r"\_", "_"))
    return f'<pre class="equation">{safe}</pre>'


def _parse_table_tex(path: Path) -> str:
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
            _clean_inline(cell).replace("~", " ").replace(r"\textwidth", "textwidth")
            for cell in cells
        ]
        rows.append(cleaned)

    deduped: list[list[str]] = []
    for row in rows:
        if deduped and row == deduped[-1]:
            continue
        deduped.append(row)
    if not deduped:
        return "<p><em>Table content unavailable.</em></p>"

    header = deduped[0]
    body = deduped[1:]
    thead = "".join(f"<th>{html.escape(cell)}</th>" for cell in header)
    body_rows = []
    for row in body:
        cells = "".join(f"<td>{html.escape(cell)}</td>" for cell in row)
        body_rows.append(f"<tr>{cells}</tr>")
    return (
        '<div class="table-wrap"><table>'
        f"<thead><tr>{thead}</tr></thead>"
        f"<tbody>{''.join(body_rows)}</tbody>"
        "</table></div>"
    )


def _parse_references(path: Path) -> str:
    raw = _read_text(path)
    entries = re.findall(r"@\w+\{([^,]+),(.*?)\n\}", raw, flags=re.S)
    items = []
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
        items.append(f"<li>{html.escape(' '.join(parts))}</li>")
    if not items:
        return "<p><em>No references found.</em></p>"
    return "<ol class=\"references\">" + "".join(items) + "</ol>"


def _extract_braced_value(tex: str, command: str) -> str:
    match = re.search(rf"\\{command}\{{(.*?)\}}", tex, flags=re.S)
    return _clean_inline(match.group(1)) if match else ""


def _render_body(tex: str) -> str:
    body = tex.split(r"\begin{document}", 1)[1].split(r"\end{document}", 1)[0]
    lines = body.splitlines()
    html_parts: list[str] = []
    paragraph_lines: list[str] = []
    current_section = ""
    idx = 0

    def flush_paragraph() -> None:
        nonlocal paragraph_lines
        text = _clean_inline(" ".join(paragraph_lines))
        paragraph_lines = []
        if text:
            html_parts.append(f"<p>{html.escape(text)}</p>")

    while idx < len(lines):
        line = lines[idx].rstrip()
        stripped = line.strip()

        if not stripped:
            flush_paragraph()
            idx += 1
            continue

        if stripped in {r"\maketitle", r"\appendix", r"\begin{landscape}", r"\end{landscape}"}:
            flush_paragraph()
            if stripped == r"\appendix":
                html_parts.append('<div class="appendix-break"></div><h2>Appendix</h2>')
            idx += 1
            continue

        if stripped == r"\begin{abstract}":
            flush_paragraph()
            abstract_lines: list[str] = []
            idx += 1
            while idx < len(lines) and lines[idx].strip() != r"\end{abstract}":
                abstract_lines.append(lines[idx].strip())
                idx += 1
            abstract_text = _clean_inline(" ".join(abstract_lines))
            html_parts.append("<h2>Abstract</h2>")
            html_parts.append(f"<p class=\"abstract\">{html.escape(abstract_text)}</p>")
            idx += 1
            continue

        section_match = re.match(r"\\section\{(.*)\}", stripped)
        if section_match:
            flush_paragraph()
            current_section = _clean_inline(section_match.group(1))
            html_parts.append(f"<h2>{html.escape(current_section)}</h2>")
            idx += 1
            continue

        subsection_match = re.match(r"\\subsection\{(.*)\}", stripped)
        if subsection_match:
            flush_paragraph()
            html_parts.append(f"<h3>{html.escape(_clean_inline(subsection_match.group(1)))}</h3>")
            idx += 1
            continue

        if stripped in {r"\begin{itemize}", r"\begin{enumerate}"}:
            flush_paragraph()
            tag = "ol" if "enumerate" in stripped else "ul"
            items: list[str] = []
            idx += 1
            while idx < len(lines):
                current = lines[idx].strip()
                if current in {r"\end{itemize}", r"\end{enumerate}"}:
                    break
                if current.startswith(r"\item"):
                    items.append(_clean_inline(current[len(r"\item") :].strip()))
                idx += 1
            rendered_items = "".join(f"<li>{html.escape(item)}</li>" for item in items if item)
            html_parts.append(f"<{tag}>{rendered_items}</{tag}>")
            idx += 1
            continue

        if stripped == r"\[":
            flush_paragraph()
            eq_lines: list[str] = []
            idx += 1
            while idx < len(lines) and lines[idx].strip() != r"\]":
                eq_lines.append(lines[idx])
                idx += 1
            html_parts.append(_equation_to_html(eq_lines))
            idx += 1
            continue

        if stripped == r"\begin{table}[H]":
            flush_paragraph()
            caption = ""
            input_path = None
            idx += 1
            while idx < len(lines) and lines[idx].strip() != r"\end{table}":
                current = lines[idx].strip()
                caption_match = re.match(r"\\caption\{(.*)\}", current)
                if caption_match:
                    caption = _clean_inline(caption_match.group(1))
                input_match = re.search(r"input\{([^}]+)\}", current)
                if input_match:
                    input_path = REPORT_ROOT / input_match.group(1)
                idx += 1
            if caption:
                html_parts.append(f"<h4>{html.escape(caption)}</h4>")
            if input_path is not None:
                html_parts.append(_parse_table_tex(input_path))
            idx += 1
            continue

        if stripped == r"\begin{figure}[H]":
            flush_paragraph()
            caption = ""
            image_path = None
            idx += 1
            while idx < len(lines) and lines[idx].strip() != r"\end{figure}":
                current = lines[idx].strip()
                image_match = re.search(r"includegraphics(?:\[[^\]]*\])?\{([^}]+)\}", current)
                if image_match:
                    image_path = image_match.group(1)
                caption_match = re.match(r"\\caption\{(.*)\}", current)
                if caption_match:
                    caption = _clean_inline(caption_match.group(1))
                idx += 1
            if image_path:
                html_parts.append(
                    '<figure>'
                    f'<img src="{html.escape(image_path)}" alt="{html.escape(caption or image_path)}" />'
                    f"<figcaption>{html.escape(caption)}</figcaption>"
                    "</figure>"
                )
            idx += 1
            continue

        input_match = re.match(r"\\input\{([^}]+)\}", stripped)
        if input_match:
            flush_paragraph()
            table_path = REPORT_ROOT / input_match.group(1)
            if current_section:
                html_parts.append(f"<h4>{html.escape(current_section)}</h4>")
            html_parts.append(_parse_table_tex(table_path))
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
    return "\n".join(html_parts)


def _build_html() -> str:
    tex = _read_text(MAIN_TEX)
    title = _extract_braced_value(tex, "title")
    author = _extract_braced_value(tex, "author")
    date = _extract_braced_value(tex, "date")
    body_html = _render_body(tex)
    refs_html = _parse_references(REFERENCES_BIB)
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <title>{html.escape(title)}</title>
  <style>
    @page {{
      size: A4;
      margin: 18mm 16mm 18mm 16mm;
    }}
    body {{
      font-family: Georgia, "Times New Roman", serif;
      color: #111;
      line-height: 1.45;
      font-size: 12px;
      margin: 0;
    }}
    .title {{
      text-align: center;
      margin-bottom: 18px;
    }}
    .title h1 {{
      font-size: 24px;
      margin: 0 0 8px 0;
      line-height: 1.2;
    }}
    .title p {{
      margin: 2px 0;
    }}
    h2 {{
      margin: 24px 0 8px;
      font-size: 18px;
      page-break-after: avoid;
    }}
    h3 {{
      margin: 18px 0 6px;
      font-size: 14px;
      page-break-after: avoid;
    }}
    h4 {{
      margin: 14px 0 6px;
      font-size: 13px;
      page-break-after: avoid;
    }}
    p, ul, ol {{
      margin: 0 0 10px 0;
    }}
    ul, ol {{
      padding-left: 20px;
    }}
    .abstract {{
      border-top: 1px solid #bbb;
      border-bottom: 1px solid #bbb;
      padding: 10px 0;
    }}
    .equation {{
      background: #f7f7f7;
      border: 1px solid #ddd;
      padding: 8px 10px;
      white-space: pre-wrap;
      font-family: "Courier New", monospace;
      font-size: 11px;
      margin: 10px 0;
    }}
    .table-wrap {{
      overflow-x: auto;
      margin: 8px 0 14px 0;
    }}
    table {{
      width: 100%;
      border-collapse: collapse;
      table-layout: fixed;
      font-size: 10.5px;
    }}
    th, td {{
      border: 1px solid #ccc;
      padding: 4px 5px;
      vertical-align: top;
      word-break: break-word;
    }}
    th {{
      background: #efefef;
      font-weight: 600;
    }}
    figure {{
      margin: 12px 0 18px;
      page-break-inside: avoid;
    }}
    figure img {{
      max-width: 100%;
      border: 1px solid #ddd;
    }}
    figcaption {{
      font-size: 11px;
      color: #444;
      margin-top: 6px;
      text-align: center;
    }}
    .appendix-break {{
      page-break-before: always;
    }}
    .references {{
      padding-left: 20px;
    }}
  </style>
</head>
<body>
  <div class="title">
    <h1>{html.escape(title)}</h1>
    <p>{html.escape(author)}</p>
    <p>{html.escape(date)}</p>
  </div>
  {body_html}
  <h2>References</h2>
  {refs_html}
</body>
</html>
"""


def _find_browser() -> Path:
    candidates = [
        Path(r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"),
        Path(r"C:\Program Files\Microsoft\Edge\Application\msedge.exe"),
        Path(r"C:\Program Files\Google\Chrome\Application\chrome.exe"),
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError("No supported browser executable found for headless PDF rendering.")


def render_pdf() -> None:
    html_text = _build_html()
    MAIN_HTML.write_text(html_text, encoding="utf-8")
    browser = _find_browser()
    cmd = [
        str(browser),
        "--headless=new",
        "--disable-gpu",
        "--allow-file-access-from-files",
        "--virtual-time-budget=10000",
        f"--print-to-pdf={MAIN_PDF}",
        MAIN_HTML.resolve().as_uri(),
    ]
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    render_pdf()
