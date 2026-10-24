"""A weekly report in Markdown as a clean page, opened in the browser.

From the page: select all and paste into an email (the tables stay tables), or
Print -> Save as PDF to attach.

  python scripts/render_report.py docs/weekly_reports/week9.md
"""

import subprocess
import sys
from pathlib import Path

import markdown

STYLE = """
body { font-family: -apple-system, Helvetica, Arial, sans-serif; font-size: 14px;
       line-height: 1.5; color: #222; background: #fff; max-width: 720px; margin: 32px auto; padding: 0 16px; }
h2 { font-size: 17px; margin: 24px 0 8px; }
table { border-collapse: collapse; margin: 6px 0 16px; }
th, td { border: 1px solid #ccc; padding: 4px 12px; text-align: left; }
th { background: #f3f3f3; }
td:not(:first-child), th:not(:first-child) { text-align: right; }
"""

src = Path(sys.argv[1])
body = markdown.markdown(src.read_text(), extensions=["tables"])
out = src.with_suffix(".html")
out.write_text(f"<!doctype html><meta charset='utf-8'><title>{src.stem}</title>"
               f"<style>{STYLE}</style>{body}\n")
print(out)
subprocess.run(["open", str(out)])
