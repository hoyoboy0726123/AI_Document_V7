"""Debug why _html_table_to_markdown is returning empty for real PPStructure output."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from app.services.ocr_pipeline import _html_table_to_markdown, _HTMLTableParser

# Sample shapes that PPStructureV3 emits (from chunks we observed)
samples = [
    # Plain <table>
    "<table><tr><th>項目</th><th>值</th></tr><tr><td>頻率</td><td>100MHz</td></tr></table>",
    # Wrapped in <html><body>
    "<html><body><table><tbody><tr><td>EMI Receiver</td><td>Agilent</td><td>N9038A</td></tr></tbody></table></body></html>",
    # With <thead>
    "<html><body><table><thead><tr><td>Mk.</td><td>Ho.</td></tr></thead><tbody><tr><td>1</td><td>2</td></tr></tbody></table></body></html>",
]

for i, html in enumerate(samples):
    print(f"\n=== Sample {i} ===")
    print(f"IN:  {html[:120]}{'...' if len(html) > 120 else ''}")
    parser = _HTMLTableParser()
    parser.feed(html)
    parser.close()
    print(f"  parsed rows: {len(parser.rows)}")
    for r in parser.rows[:3]:
        print(f"    row: {r}")
    md = _html_table_to_markdown(html)
    print(f"OUT (md):")
    print(md if md else "  <EMPTY>")
