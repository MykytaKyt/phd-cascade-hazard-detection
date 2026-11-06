#!/usr/bin/env python3
"""Assemble the dissertation draft into one .docx.

Order: title page, АНОТАЦІЯ / SUMMARY, ЗМІСТ, ВСТУП, розділи 1-5, ВИСНОВКИ, СПИСОК
ВИКОРИСТАНИХ ДЖЕРЕЛ, ДОДАТОК А. Citations [key; key] become numbers in order of
first citation; the reference list is formatted after ДСТУ 8302:2015 (simplified).
Everything that still has to be filled in ([уточнити], [N], [рік], author's papers
without output data, missing figures, bibliography records with an open `todo`)
is highlighted in yellow and prefixed with TODO.

Formulas: LaTeX display math with \\tag{N} becomes a centred formula with the number
(N) at the right margin.

Usage:
    python tools/build_docx.py --out build/dissertation_draft.docx
"""
import argparse
import re
import subprocess
import tempfile
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
THESIS = ROOT / "docs" / "thesis"
ORDER = ["intro", "ch1", "ch2", "ch3", "ch4", "ch5", "conclusions"]
# author's own papers cited by short labels; output data still to be filled in
OWN = {"iJOE": "стаття iJOE", "UzhNU": "стаття УжНУ", "PCI": "стаття PCI"}
ALIASES = {"SeePhys": "xiang2025seephys"}
TEXT_WIDTH = 9638  # A4, left 25 mm, right 15 mm (department template), in twips

TODO_WORDS = r"уточнити|to be confirmed|year\b|\bрік\b|За наявності|Видання|Тези|Обсяг|після оформлення|^N$"


# ---------------------------------------------------------------- bibliography
def parse_bib(paths):
    entries = {}
    for path in paths:
        text = path.read_text(encoding="utf-8")
        for m in re.finditer(r"@(\w+)\s*\{\s*([^,\s]+)\s*,", text):
            kind, key = m.group(1).lower(), m.group(2)
            i, depth = m.end(), 1
            while depth and i < len(text):
                depth += {"{": 1, "}": -1}.get(text[i], 0)
                i += 1
            body = text[m.end():i - 1]
            fields, j = {}, 0
            for fm in re.finditer(r"(\w+)\s*=\s*", body):
                if fm.start() < j:
                    continue
                k = fm.end()
                if body[k] == "{":
                    d, e = 1, k + 1
                    while d and e < len(body):
                        d += {"{": 1, "}": -1}.get(body[e], 0)
                        e += 1
                    val = body[k + 1:e - 1]
                else:
                    e = body.find(",", k)
                    e = len(body) if e < 0 else e
                    val = body[k:e]
                j = e
                fields[fm.group(1).lower()] = re.sub(r"\s+", " ", delatex(val).replace("{", "").replace("}", "")).strip()
            entries[key] = dict(kind=kind, **fields)
    return entries


ACCENTS = {"'": "\u0301", "`": "\u0300", '"': "\u0308", "~": "\u0303", "^": "\u0302"}


def delatex(v: str) -> str:
    import unicodedata
    v = re.sub(r"\{?\\([`'\"~^])\{?([A-Za-z])\}?\}?", lambda m: unicodedata.normalize("NFC", m.group(2) + ACCENTS[m.group(1)]), v)
    v = re.sub(r"\{?\\(oe|ae|o|ss)\}?", lambda m: {"oe": "œ", "ae": "æ", "o": "ø", "ss": "ß"}[m.group(1)], v)
    return v.replace("--", "–").replace("\\&", "&")


def split_name(name: str) -> tuple[str, str]:
    """(surname, initials) of a BibTeX name: 'Last, First Middle' or 'First Middle Last'."""
    name = name.strip()
    if "," in name:
        last, first = [x.strip() for x in name.split(",", 1)]
    else:
        parts = name.split()
        last, first = parts[-1], " ".join(parts[:-1])
    initials = " ".join(p[0] + "." for p in re.split(r"[\s-]+", first) if p and p[0].isalpha())
    return last, initials


def format_entry(e: dict) -> str:
    """Bibliographic description after ДСТУ 8302:2015, as in the samples of the department:
    1–3 authors: 'Прізвище І. Б., Прізвище І. Б. Назва // Журнал. Рік. Т. 1, № 2. С. 3–4.';
    4 authors: 'Назва / І. Б. Прізвище, ... // ...'; 5 and more: the first three and [та ін.]."""
    cyr = bool(re.search(r"[А-Яа-яІіЇїЄєҐґ]", e.get("title", "")))
    names = [split_name(a) for a in e.get("author", "").split(" and ") if a.strip()]
    etal = any(l.lower() == "others" for l, _ in names)
    names = [(l, i) for l, i in names if l.lower() != "others"]
    if etal and len(names) < 5:
        names = names + [("", "")] * (5 - len(names))  # treat as five or more authors
    title = e.get("title", "").rstrip(". ")
    if 1 <= len(names) <= 3:
        head = ", ".join(f"{l} {i}".strip() for l, i in names) + " " + title
    elif len(names) >= 4:
        shown = [n for n in (names if len(names) == 4 else names[:3]) if n[0]]
        tail = "" if len(names) == 4 else (" [та ін.]" if cyr else " [et al.]")
        head = title + " / " + ", ".join(f"{i} {l}".strip() for l, i in shown) + tail
    else:
        head = title
    vol, no, pp = ("Т.", "№", "С.") if cyr else ("Vol.", "No.", "P.")
    year = e.get("year", "")
    kind = e.get("kind")
    pages = e.get("pages", "").replace("--", "–").replace("-", "–")
    if kind == "article":
        out = f"{head} // {e.get('journal', '')}. {year}."
        if e.get("volume"):
            out += f" {vol} {e['volume']}" + ("," if e.get("number") else ".")
        if e.get("number"):
            out += f" {no} {e['number']}."
        if pages:
            out += f" {pp} {pages}."
    elif kind in ("inproceedings", "incollection"):
        place = ", ".join(x for x in (e.get("address"), e.get("publisher")) if x)
        out = f"{head} // {e.get('booktitle', '')}." + (f" {place}," if place else "") + f" {year}."
        if pages:
            out += f" {pp} {pages}."
    elif kind == "book":
        place = " : ".join(x for x in (e.get("address"), e.get("publisher")) if x)
        out = f"{head}." + (f" {place}," if place else "") + f" {year}."
    elif kind == "techreport":
        out = f"{head} : {e.get('type', 'tech. rep.')} / {e.get('institution', '')}. {year}."
    elif e.get("eprint"):
        out = f"{head} // arXiv. {year}. arXiv:{e['eprint']}."
    else:
        out = f"{head}. {year}."
    if e.get("doi"):
        out += f" DOI: {e['doi']}."
    elif e.get("url"):
        v = e.get("verified", "")
        date = ".".join(reversed(v.split("-"))) if re.match(r"\d{4}-\d\d-\d\d$", v) else ""
        out += f" URL: {e['url']}" + (f" (дата звернення: {date})." if date else ".")
    return re.sub(r"\s+", " ", out).replace(" .", ".")


# ---------------------------------------------------------------- markdown
def todo(text: str) -> str:
    text = text.strip().strip("[]").replace("[", "(").replace("]", ")")
    return f'[TODO: {text}]{{custom-style="TODO"}}'


def number_formulas(md: str) -> str:
    def repl(m):
        body = m.group(1)
        tag = re.search(r"\\tag\{([^}]*)\}", body)
        if not tag:
            return m.group(0)
        body = re.sub(r"\\tag\{[^}]*\}", "", body).strip()
        body = " ".join(line.strip() for line in body.splitlines())
        return f'\n::: {{custom-style="Formula"}}\nQQTABQQ${body}$QQTABQQ({tag.group(1)})\n:::\n'
    return re.sub(r"\$\$(.*?)\$\$", repl, md, flags=re.S)


def cite(md: str, bib: dict, order: list) -> str:
    def num(key):
        if key not in order:
            order.append(key)
        return order.index(key) + 1

    def repl(m):
        keys = [k.strip() for k in re.split(r"[;,]", m.group(1))]
        keys = [ALIASES.get(k, k) for k in keys]
        if all(k in OWN for k in keys):
            return todo("посилання: " + ", ".join(OWN[k] for k in keys) + " — вихідні дані")
        if not all(k in bib for k in keys):
            return m.group(0)
        return "[" + ", ".join(str(num(k)) for k in keys) + "]"
    return re.sub(r"(?<![\]\w$\\])\[([A-Za-z][\w-]*(?:\s*[;,]\s*[A-Za-z][\w-]*)*)\](?![(\[{])", repl, md)


def prepare(md: str, bib: dict, order: list) -> str:
    md = re.sub(r"<!--.*?-->", "", md, flags=re.S)
    md = number_formulas(md)
    # placeholders outside math
    pieces = re.split(r"(\$[^$]*\$)", md)
    for i in range(0, len(pieces), 2):
        p = pieces[i]
        p = re.sub(r"\[(?:[^\[\]]*?)(?:" + TODO_WORDS + r")[^\[\]]*\]", lambda m: todo(m.group(0)), p)
        p = re.sub(r"\[N\]", lambda m: todo("N"), p)
        p = re.sub(r"рис\. \d+\.x", lambda m: todo(m.group(0) + " — номер рисунка"), p)
        p = re.sub(r"(?m)^(Рисунок \d+\.\d+ — .*)$", lambda m: todo("вставити рисунок") + "\n\n" + m.group(1), p)
        p = re.sub(r"(Рисунок|Таблиця|Таблица) (\d+\.\d+) — ", r"\1 \2 – ", p)
        p = re.sub(r"(?m)^(Таблиця \d+\.\d+ – (?:.+\n)*?)(?=\n)",
                   lambda m: '::: {custom-style="Table Caption"}\n' + m.group(1) + ":::\n", p)
        pieces[i] = p
    md = "".join(pieces)
    return cite(md, bib, order)


TITLE = """
::: {custom-style="Center"}
Міністерство освіти і науки України

Харківський національний університет радіоелектроніки
:::

::: {custom-style="Right"}
Кваліфікаційна наукова праця на правах рукопису

УДК 004.93:519.8 [TODO: перевірити УДК дисертації]{custom-style="TODO"}
:::

::: {custom-style="Center"}
**КИТ МИКИТА ОЛЕКСАНДРОВИЧ** [TODO: перевірити по батькові]{custom-style="TODO"}

**ДОСЛІДЖЕННЯ ТА РОЗРОБКА МЕТОДІВ ГЛИБОКОГО НАВЧАННЯ ДЛЯ ВИЯВЛЕННЯ НЕБЕЗПЕЧНИХ ПОДІЙ НА ОСНОВІ ЗОРОВОГО СПОСТЕРЕЖЕННЯ**

Спеціальність [TODO: 113 Прикладна математика — уточнити]{custom-style="TODO"}

Галузь знань [TODO: 11 Математика та статистика — уточнити]{custom-style="TODO"}

Подається на здобуття наукового ступеня доктора філософії
:::

Дисертація містить результати власних досліджень. Використання ідей, результатів і текстів інших авторів мають
посилання на відповідне джерело.

______________ М. О. Кит

::: {custom-style="Right"}
Науковий керівник: Єсілевський Валентин Сергійович, кандидат технічних наук, доцент
:::

::: {custom-style="Center"}
Харків — [TODO: рік]{custom-style="TODO"}
:::
"""


def build_markdown() -> str:
    bib = parse_bib([ROOT / "references.bib", ROOT / "references_extra.bib"])
    order = []
    abstract = (THESIS / "abstract.md").read_text(encoding="utf-8")
    abstract, publications = abstract.split("## Список публікацій здобувача", 1)
    pubs, summary = publications.split("---", 1)
    body = [TITLE, prepare(abstract, bib, order),
            "## Список публікацій здобувача" + prepare(pubs, bib, order), prepare(summary, bib, order)]
    body.append("# ЗМІСТ\n\n" + todo("оновити поле змісту у Word: правою кнопкою → Оновити поле") +
                '\n\n```{=openxml}\n<w:p><w:r><w:fldChar w:fldCharType="begin" w:dirty="true"/></w:r>'
                '<w:r><w:instrText xml:space="preserve"> TOC \\o "1-3" \\h \\z \\u </w:instrText></w:r>'
                '<w:r><w:fldChar w:fldCharType="separate"/></w:r><w:r><w:t>Зміст</w:t></w:r>'
                '<w:r><w:fldChar w:fldCharType="end"/></w:r></w:p>\n```\n')
    for name in ORDER:
        body.append(prepare((THESIS / f"{name}.md").read_text(encoding="utf-8"), bib, order))
    refs = ["# СПИСОК ВИКОРИСТАНИХ ДЖЕРЕЛ", "",
            todo("звірити оформлення з ДСТУ 8302:2015 і вимогами кафедри; додати публікації здобувача"), ""]
    for n, key in enumerate(order, 1):
        e = bib[key]
        line = format_entry(e).replace("*", r"\*")
        if e.get("todo"):
            line += " " + todo("перевірити запис: " + e["todo"][:160])
        refs.append(f"{n}\\. {line}\n")
    body.append("\n".join(refs))
    body.append("# ДОДАТОК А\n\n## Список публікацій здобувача за темою дисертації та відомості про апробацію\n\n" +
                todo("заповнити за docs/publications.md: вихідні дані, DOI, індексація; конференції — назва, місто, дата"))
    return "\n\n".join(body)


# ---------------------------------------------------------------- docx styling
def _fonts(style_or_run_el, size=None):
    """Times New Roman for all scripts (and size in half-points) on an rPr owner."""
    from docx.oxml.ns import qn
    from docx.oxml import OxmlElement
    rpr = style_or_run_el.get_or_add_rPr()
    fonts = rpr.find(qn("w:rFonts"))
    if fonts is None:
        fonts = OxmlElement("w:rFonts")
        rpr.insert(0, fonts)
    for k in list(fonts.attrib):
        del fonts.attrib[k]
    for a in ("w:ascii", "w:hAnsi", "w:eastAsia", "w:cs"):
        fonts.set(qn(a), "Times New Roman")
    if size:
        for tag in ("w:sz", "w:szCs"):
            el = rpr.find(qn(tag))
            if el is None:
                el = OxmlElement(tag)
                rpr.append(el)
            el.set(qn("w:val"), str(size))


def style_reference(path: Path):
    """Pandoc's reference.docx restyled for the dissertation: Times New Roman 14 everywhere,
    1.5 spacing, 1.25 cm indent, black headings, bordered tables, Ukrainian language."""
    from docx import Document
    from docx.enum.style import WD_STYLE_TYPE
    from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_LINE_SPACING, WD_TAB_ALIGNMENT
    from docx.oxml import OxmlElement
    from docx.oxml.ns import qn
    from docx.shared import Cm, Pt, RGBColor, Twips

    ref = subprocess.run(["pandoc", "--print-default-data-file", "reference.docx"],
                         capture_output=True, check=True).stdout
    path.write_bytes(ref)
    d = Document(str(path))
    styles = d.styles
    # document defaults: font, size 14, Ukrainian
    defaults = styles.element.find(qn("w:docDefaults"))
    rpr_def = defaults.find(qn("w:rPrDefault")).find(qn("w:rPr"))
    for el in list(rpr_def):
        rpr_def.remove(el)
    for xml in ('<w:rFonts %s w:ascii="Times New Roman" w:hAnsi="Times New Roman" w:eastAsia="Times New Roman" '
                'w:cs="Times New Roman"/>', '<w:sz %s w:val="28"/>', '<w:szCs %s w:val="28"/>',
                '<w:lang %s w:val="uk-UA" w:eastAsia="uk-UA" w:bidi="ar-SA"/>'):
        from docx.oxml import parse_xml
        rpr_def.append(parse_xml(xml % 'xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main"'))

    def para(name, size=14, indent=1.25, align=WD_ALIGN_PARAGRAPH.JUSTIFY, spacing=1.5, before=0, after=0,
             bold=None, italic=False):
        found = [x for x in styles if x.name == name]
        st = found[0] if found else styles.add_style(name, WD_STYLE_TYPE.PARAGRAPH)
        _fonts(st.element, size * 2)
        st.font.color.rgb = RGBColor(0, 0, 0)
        st.font.italic = italic
        if bold is not None:
            st.font.bold = bold
        pf = st.paragraph_format
        pf.first_line_indent = Cm(indent)
        pf.alignment = align
        pf.line_spacing = spacing
        pf.space_before, pf.space_after = Pt(before), Pt(after)
        return st

    for name in ("Normal", "Body Text", "First Paragraph", "Block Text", "Footnote Text", "Bibliography"):
        if name in [x.name for x in styles]:
            para(name)
    para("Compact")  # tight lists in the text: as body text
    para("Table Text", size=12, indent=0, align=WD_ALIGN_PARAGRAPH.LEFT, spacing=1.0)
    for name, level in (("Heading 1", 1), ("Heading 2", 2), ("Heading 3", 3), ("Heading 4", 4)):
        st = para(name, indent=0 if level == 1 else 1.25,
                  align=WD_ALIGN_PARAGRAPH.CENTER if level == 1 else WD_ALIGN_PARAGRAPH.LEFT,
                  spacing=1.5, before=42 if level > 1 else 0, after=21, bold=level == 1)
        st.paragraph_format.keep_with_next = True
        if level == 1:
            st.paragraph_format.page_break_before = True
    for name in ("Title", "Subtitle", "Author", "Date", "Abstract"):
        if name in [x.name for x in styles]:
            para(name, indent=0, align=WD_ALIGN_PARAGRAPH.CENTER)
    # department template: figure caption centred under the figure, an empty line around;
    # table caption above the table from the paragraph indent
    for name in ("Caption", "Image Caption"):
        if name in [x.name for x in styles]:
            para(name, indent=0, align=WD_ALIGN_PARAGRAPH.CENTER, spacing=1.5, before=6, after=21)
    st = para("Table Caption", indent=1.25, align=WD_ALIGN_PARAGRAPH.LEFT, spacing=1.5, before=21, after=0)
    st.paragraph_format.keep_with_next = True
    for name in ("Figure", "Captioned Figure"):
        if name in [x.name for x in styles]:
            st = para(name, indent=0, align=WD_ALIGN_PARAGRAPH.CENTER, spacing=1.0, before=21)
            st.paragraph_format.keep_with_next = True
    para("Center", indent=0, align=WD_ALIGN_PARAGRAPH.CENTER)
    st = para("Right", indent=0, align=WD_ALIGN_PARAGRAPH.LEFT)
    st.paragraph_format.left_indent = Cm(9)
    st = para("Formula", indent=0, align=WD_ALIGN_PARAGRAPH.LEFT, before=6, after=6)
    st.paragraph_format.tab_stops.add_tab_stop(Twips(TEXT_WIDTH // 2), WD_TAB_ALIGNMENT.CENTER)
    st.paragraph_format.tab_stops.add_tab_stop(Twips(TEXT_WIDTH), WD_TAB_ALIGNMENT.RIGHT)
    # character styles: inline code in the text font, TODO highlighted
    for st in styles:
        if st.type == WD_STYLE_TYPE.CHARACTER:
            _fonts(st.element)
            st.font.color.rgb = RGBColor(0, 0, 0)
    todo_st = styles.add_style("TODO", WD_STYLE_TYPE.CHARACTER)
    hl = OxmlElement("w:highlight")
    hl.set(qn("w:val"), "yellow")
    todo_st.element.get_or_add_rPr().append(hl)
    # tables: all borders, single spacing
    for st in styles:
        if st.type == WD_STYLE_TYPE.TABLE:
            tpr = st.element.find(qn("w:tblPr"))
            if tpr is None:
                tpr = OxmlElement("w:tblPr")
                st.element.append(tpr)
            old = tpr.find(qn("w:tblBorders"))
            if old is not None:
                tpr.remove(old)
            borders = OxmlElement("w:tblBorders")
            for edge in ("top", "left", "bottom", "right", "insideH", "insideV"):
                e = OxmlElement(f"w:{edge}")
                for k, v in (("val", "single"), ("sz", "4"), ("space", "0"), ("color", "000000")):
                    e.set(qn(f"w:{k}"), v)
                borders.append(e)
            tpr.append(borders)
    d.save(str(path))


def postprocess(path: Path):
    tmp = path.with_suffix(".tmp.docx")
    with zipfile.ZipFile(path) as zin, zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED) as zout:
        for item in zin.infolist():
            data = zin.read(item.filename)
            if item.filename == "word/document.xml":
                x = data.decode("utf-8")
                x = re.sub(r"QQTABQQ", '</w:t><w:tab/><w:t xml:space="preserve">', x)
                x = re.sub(r'<w:t xml:space="preserve"></w:t>', "", x)
                x = re.sub(r'<w:t></w:t>', "", x)
                # table cells: smaller single-spaced text
                x = re.sub(r"<w:tbl>.*?</w:tbl>", lambda m: m.group(0).replace(
                    '<w:pStyle w:val="Compact" />', '<w:pStyle w:val="TableText" />').replace(
                    '<w:pStyle w:val="Compact"/>', '<w:pStyle w:val="TableText"/>'), x, flags=re.S)
                x = x.replace('<w:rStyle w:val="TODO" />', '<w:rStyle w:val="TODO" /><w:highlight w:val="yellow" />')
                x = x.replace('<w:rStyle w:val="TODO"/>', '<w:rStyle w:val="TODO"/><w:highlight w:val="yellow"/>')
                data = x.encode("utf-8")
            elif item.filename == "word/numbering.xml":
                x = data.decode("utf-8")
                x = re.sub(r'(<w:numFmt w:val="bullet" ?/>\s*<w:lvlText w:val=")[^"]*(")', r"\1–\2", x)
                data = x.encode("utf-8")
            zout.writestr(item, data)
    tmp.replace(path)
    page_setup(path)


def page_setup(path: Path):
    """A4, margins of the department template (25/15/20/20 mm), page number at the top right,
    none on the title page."""
    from docx import Document
    from docx.enum.text import WD_ALIGN_PARAGRAPH
    from docx.oxml import OxmlElement
    from docx.oxml.ns import qn
    from docx.shared import Mm, Pt
    d = Document(str(path))
    for sec in d.sections:
        sec.page_width, sec.page_height = Mm(210), Mm(297)
        sec.left_margin, sec.right_margin = Mm(25), Mm(15)
        sec.top_margin, sec.bottom_margin = Mm(20), Mm(20)
        sec.header_distance, sec.footer_distance = Mm(12.5), Mm(12.5)
        sec.different_first_page_header_footer = True
        par = sec.header.paragraphs[0]
        par.alignment = WD_ALIGN_PARAGRAPH.RIGHT
        par.paragraph_format.first_line_indent = Pt(0)
        run = par.add_run()
        for tag, attr in (("w:fldChar", "begin"), ("w:instrText", None), ("w:fldChar", "end")):
            el = OxmlElement(tag)
            if attr:
                el.set(qn("w:fldCharType"), attr)
            else:
                el.set(qn("xml:space"), "preserve")
                el.text = " PAGE "
            run._r.append(el)
    d.save(str(path))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", default=ROOT / "build" / "dissertation_draft.docx")
    parser.add_argument("--keep-md", action="store_true")
    args = parser.parse_args()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    md = build_markdown()
    with tempfile.TemporaryDirectory() as d:
        mdp, ref = Path(d) / "thesis.md", Path(d) / "reference.docx"
        mdp.write_text(md, encoding="utf-8")
        if args.keep_md:
            (out.parent / "dissertation_draft.md").write_text(md, encoding="utf-8")
        style_reference(ref)
        subprocess.run(["pandoc", str(mdp), "-f", "markdown+pipe_tables+tex_math_dollars", "-o", str(out),
                        "--reference-doc", str(ref), "--resource-path", str(ROOT)], check=True)
    postprocess(out)
    print(f"Saved {out}; TODO markers: {md.count('[TODO:')}")
    # short summary for the supervisor, same styles
    summary = out.parent / "summary.docx"
    with tempfile.TemporaryDirectory() as d:
        mdp, ref = Path(d) / "summary.md", Path(d) / "reference.docx"
        text = re.sub(r"<!--.*?-->", "", (THESIS / "summary.md").read_text(encoding="utf-8"), flags=re.S)
        mdp.write_text(text.replace("# РОЗШИРЕНЕ РЕЗЮМЕ", "::: {custom-style=\"Center\"}\n**РОЗШИРЕНЕ РЕЗЮМЕ", 1)
                       .replace("ДИСЕРТАЦІЇ\n", "ДИСЕРТАЦІЇ**\n:::\n", 1), encoding="utf-8")
        style_reference(ref)
        subprocess.run(["pandoc", str(mdp), "-f", "markdown+pipe_tables+tex_math_dollars", "-o", str(summary),
                        "--reference-doc", str(ref)], check=True)
    postprocess(summary)
    print(f"Saved {summary}")


if __name__ == "__main__":
    main()
