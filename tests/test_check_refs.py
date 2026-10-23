import importlib.util
from pathlib import Path

spec = importlib.util.spec_from_file_location(
    "check_refs", Path(__file__).resolve().parents[1] / "tools" / "check_refs.py"
)
check_refs = importlib.util.module_from_spec(spec)
spec.loader.exec_module(check_refs)

BIB = """
@article{hof2005,
  author = {Hof, A. L. and Gazendam, M. G. J. and Sinke, W. E.},
  title = {The condition for dynamic {stability}},
  journal = {Journal of Biomechanics},
  year = 2005,
  doi = {10.1016/j.jbiomech.2004.03.025},
  verified = "2026-10-23",
}

@misc{ucfcrime, title = {Real-world anomaly detection}, url = {https://example.org/a,b}}
"""


def test_parse_bib_fields():
    entries = check_refs.parse_bib(BIB)
    assert [e["ID"] for e in entries] == ["hof2005", "ucfcrime"]
    hof = entries[0]
    assert hof["ENTRYTYPE"] == "article"
    assert hof["title"] == "The condition for dynamic {stability}"
    assert hof["year"] == "2005"
    assert hof["doi"] == "10.1016/j.jbiomech.2004.03.025"
    assert hof["verified"] == "2026-10-23"
    assert entries[1]["url"] == "https://example.org/a,b"


def test_normalise_ignores_braces_and_case():
    assert check_refs.normalise("The {Condition} for Dynamic Stability.") == "the condition for dynamic stability"
