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
TEXT_WIDTH = 9921  # A4, left 25 mm, right 10 mm, in twips

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


def short_name(name: str) -> str:
    name = name.strip()
    if "," in name:
        last, first = [x.strip() for x in name.split(",", 1)]
    else:
        parts = name.split()
        last, first = parts[-1], " ".join(parts[:-1])
    initials = " ".join(p[0] + "." for p in re.split(r"[\s-]+", first) if p)
    return f"{last} {initials}".strip()


def format_entry(e: dict) -> str:
    authors = [short_name(a) for a in e.get("author", "").split(" and ") if a.strip()]
    cyr = bool(re.search(r"[А-Яа-яІіЇїЄєҐґ]", e.get("title", "")))
    head = ""
    if authors:
        head = (", ".join(authors[:3]) + (" та ін." if cyr else " et al.") if len(authors) > 3
                else ", ".join(authors)) + " "
    parts = [head + e.get("title", "").rstrip(".") + "."]
    venue = e.get("journal") or e.get("booktitle") or e.get("publisher") or e.get("institution") or ""
    if e.get("kind") in ("misc",) and e.get("eprint"):
        venue = f"arXiv:{e['eprint']}"
    if venue:
        parts.append(venue + ".")
    if e.get("address") and e.get("kind") == "book":
        parts[-1] = f"{e['address']} : {venue}."
    parts.append(e.get("year", "") + ".")
    if e.get("volume"):
        parts.append(("Т. " if cyr else "Vol. ") + e["volume"] + (", " if e.get("number") else "."))
    if e.get("number"):
        parts.append(("№ " if cyr else "no. ") + e["number"] + ".")
    if e.get("pages"):
        parts.append(("С. " if cyr else "P. ") + e["pages"].replace("--", "–") + ".")
    if e.get("doi"):
        parts.append("DOI: " + e["doi"] + ".")
    elif e.get("url"):
        parts.append("URL: " + e["url"] + ".")
    return " ".join(p for p in parts if p.strip(". "))


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

[TODO: УДК]{custom-style="TODO"}
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
Науковий керівник: Єсілевський Валентин Сергійович, [TODO: науковий ступінь, вчене звання]{custom-style="TODO"}
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
STYLE_XML = """
<w:style w:type="character" w:customStyle="1" w:styleId="TODO"><w:name w:val="TODO"/>
  <w:rPr><w:highlight w:val="yellow"/></w:rPr></w:style>
<w:style w:type="paragraph" w:customStyle="1" w:styleId="Formula"><w:name w:val="Formula"/><w:basedOn w:val="Normal"/>
  <w:pPr><w:tabs><w:tab w:val="center" w:pos="{c}"/><w:tab w:val="right" w:pos="{r}"/></w:tabs>
  <w:spacing w:before="120" w:after="120"/><w:ind w:firstLine="0"/><w:jc w:val="left"/></w:pPr></w:style>
<w:style w:type="paragraph" w:customStyle="1" w:styleId="Center"><w:name w:val="Center"/><w:basedOn w:val="Normal"/>
  <w:pPr><w:ind w:firstLine="0"/><w:jc w:val="center"/></w:pPr></w:style>
<w:style w:type="paragraph" w:customStyle="1" w:styleId="Right"><w:name w:val="Right"/><w:basedOn w:val="Normal"/>
  <w:pPr><w:ind w:left="5000" w:firstLine="0"/><w:jc w:val="left"/></w:pPr></w:style>
""".format(c=TEXT_WIDTH // 2, r=TEXT_WIDTH)


def style_reference(path: Path):
    ref = subprocess.run(["pandoc", "--print-default-data-file", "reference.docx"],
                         capture_output=True, check=True).stdout
    src = path.with_suffix(".src.docx")
    src.write_bytes(ref)
    with zipfile.ZipFile(src) as zin, zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as zout:
        for item in zin.infolist():
            data = zin.read(item.filename)
            if item.filename == "word/styles.xml":
                x = data.decode("utf-8")
                fonts = ('<w:rFonts w:ascii="Times New Roman" w:hAnsi="Times New Roman" '
                         'w:eastAsia="Times New Roman" w:cs="Times New Roman"/>')
                x = re.sub(r"<w:rFonts [^>]*/>", fonts, x)
                x = re.sub(r'<w:sz w:val="\d+"/>', '<w:sz w:val="28"/>', x)
                x = re.sub(r'<w:szCs w:val="\d+"/>', '<w:szCs w:val="28"/>', x)
                x = re.sub(r'<w:color w:val="[0-9A-Fa-f]+"[^>]*/>', '<w:color w:val="000000"/>', x)
                # Normal: 1.5 line spacing, first-line indent 1.25 cm, justified
                x = re.sub(r'(<w:style w:type="paragraph" w:default="1" w:styleId="Normal">.*?)(<w:rPr>)',
                           r'\1<w:pPr><w:spacing w:after="0" w:line="360" w:lineRule="auto"/>'
                           r'<w:ind w:firstLine="709"/><w:jc w:val="both"/></w:pPr>\2', x, count=1, flags=re.S)
                for sid in ("BodyText", "FirstParagraph"):
                    x = re.sub(rf'(<w:style w:type="paragraph"[^>]*w:styleId="{sid}">.*?)<w:pPr>.*?</w:pPr>',
                               r'\1<w:pPr><w:spacing w:before="0" w:after="0" w:line="360" w:lineRule="auto"/>'
                               r'<w:ind w:firstLine="709"/><w:jc w:val="both"/></w:pPr>', x, count=1, flags=re.S)
                x = re.sub(r'(<w:style w:type="paragraph"[^>]*w:styleId="Heading1">.*?<w:pPr>)',
                           r'\1<w:pageBreakBefore/><w:jc w:val="center"/>', x, count=1, flags=re.S)
                for sid in ("Heading1", "Heading2", "Heading3"):
                    x = re.sub(rf'(<w:style w:type="paragraph"[^>]*w:styleId="{sid}">.*?<w:rPr>)',
                               r'\1<w:b/><w:i w:val="0"/>', x, count=1, flags=re.S)
                # table cells and tight lists: single spacing, no indent
                x = re.sub(r'(<w:style w:type="paragraph"[^>]*w:styleId="Compact">.*?)<w:pPr>.*?</w:pPr>',
                           r'\1<w:pPr><w:spacing w:before="0" w:after="0" w:line="240" w:lineRule="auto"/>'
                           r'<w:ind w:firstLine="0"/><w:jc w:val="left"/></w:pPr>', x, count=1, flags=re.S)
                x = x.replace("</w:styles>", STYLE_XML + "</w:styles>")
                data = x.encode("utf-8")
            zout.writestr(item, data)
    src.unlink()


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
                x = x.replace('<w:rStyle w:val="TODO" />', '<w:rStyle w:val="TODO" /><w:highlight w:val="yellow" />')
                x = x.replace('<w:rStyle w:val="TODO"/>', '<w:rStyle w:val="TODO"/><w:highlight w:val="yellow"/>')
                sect = ('<w:sectPr><w:pgSz w:w="11906" w:h="16838"/><w:pgMar w:top="1134" w:right="567" '
                        'w:bottom="1134" w:left="1418" w:header="709" w:footer="709" w:gutter="0"/></w:sectPr>')
                x = re.sub(r"<w:sectPr\b.*?</w:sectPr>", sect, x, flags=re.S)
                data = x.encode("utf-8")
            zout.writestr(item, data)
    tmp.replace(path)


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


if __name__ == "__main__":
    main()
