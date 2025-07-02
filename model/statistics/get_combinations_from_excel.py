#!/usr/bin/env python3
"""Extract (source, target, model_type) combinations from the ``p-value``
worksheet of ``performance_summary.xlsx`` without heavy dependencies.

Each row corresponds to a single combination and is emitted in the same order as
the sheet.  Duplicates are kept to faithfully mirror the worksheet contents."""
import os
import re
import sys
import zipfile

EXCEL_PATH = os.path.join(os.path.dirname(__file__), '..', 'visual', 'performance_summary.xlsx')
SHEET_NAME = 'p-value'

if not os.path.exists(EXCEL_PATH):
    sys.stderr.write(f"Excel file not found: {EXCEL_PATH}\n")
    sys.exit(1)

# Helper to find the worksheet xml file corresponding to SHEET_NAME
sheet_xml = 'xl/worksheets/sheet1.xml'
try:
    with zipfile.ZipFile(EXCEL_PATH) as z:
        wb = z.read('xl/workbook.xml').decode('utf-8')
        rels = z.read('xl/_rels/workbook.xml.rels').decode('utf-8')
        rid = None
        for m in re.finditer(r'<sheet[^>]*name="([^"]+)"[^>]*r:id="([^"]+)"', wb):
            name, r = m.groups()
            if name.lower() == SHEET_NAME.lower():
                rid = r
                break
        if rid:
            m = re.search(r'<Relationship[^>]*Id="%s"[^>]*Target="([^"]+)"' % rid, rels)
            if m:
                sheet_xml = 'xl/' + m.group(1)
        data = z.read(sheet_xml).decode('utf-8')
except Exception as e:
    sys.stderr.write(f"Failed to read Excel file: {e}\n")
    sys.exit(1)

sheet_data = re.search(r'<sheetData>(.*)</sheetData>', data, re.DOTALL)
if not sheet_data:
    sys.exit(0)
rows = re.findall(r'<row[^>]*>(.*?)</row>', sheet_data.group(1), re.DOTALL)
if not rows:
    sys.exit(0)

cells = re.findall(r'<c[^>]*>(.*?)</c>', rows[0], re.DOTALL)
headers = []
for c in cells:
    m = re.search(r'<t[^>]*>(.*?)</t>', c)
    if not m:
        m = re.search(r'<v[^>]*>(.*?)</v>', c)
    headers.append(m.group(1) if m else '')

h_index = {h: i for i, h in enumerate(headers)}
required = ['source', 'target', 'model_type']
missing = [r for r in required if r not in h_index]
if missing:
    sys.stderr.write(f"Required columns missing in sheet: {missing}\n")
    sys.exit(1)

for row in rows[1:]:
    cells = re.findall(r'<c[^>]*>(.*?)</c>', row, re.DOTALL)
    values = [''] * len(headers)
    for idx, c in enumerate(cells):
        m = re.search(r'<t[^>]*>(.*?)</t>', c)
        if not m:
            m = re.search(r'<v[^>]*>(.*?)</v>', c)
        values[idx] = m.group(1) if m else ''
    record = dict(zip(headers, values))
    key = tuple(record.get(k, '') for k in required)
    if all(key):
        print('\t'.join(key))
