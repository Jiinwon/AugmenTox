#!/usr/bin/env python3
"""Extract unique (source, target, model_type) combinations from the
'p-value' sheet of performance_summary.xlsx without external
dependencies.
Prints tab-separated lines: source\ttarget\tmodel_type
"""
import os
import sys
import zipfile
import xml.etree.ElementTree as ET

# Excel file path relative to this script
EXCEL_PATH = os.path.join(os.path.dirname(__file__), '..', 'visual', 'performance_summary.xlsx')

if not os.path.exists(EXCEL_PATH):
    sys.stderr.write(f"Excel file not found: {EXCEL_PATH}\n")
    sys.exit(1)

# Namespace used in xlsx XML files
NS = {'main': 'http://schemas.openxmlformats.org/spreadsheetml/2006/main'}

with zipfile.ZipFile(EXCEL_PATH) as z:
    with z.open('xl/worksheets/sheet1.xml') as f:
        tree = ET.parse(f)

root = tree.getroot()
rows = root.findall('.//main:row', NS)
if not rows:
    sys.exit(0)

# First row contains headers
headers = []
for c in rows[0]:
    text = None
    t = c.find('.//main:t', NS)
    if t is not None:
        text = t.text
    elif c.text:
        text = c.text
    headers.append(text or '')

h_index = {name: idx for idx, name in enumerate(headers)}
required = ['source', 'target', 'model_type']
missing = [r for r in required if r not in h_index]
if missing:
    sys.stderr.write(f"Required columns missing in sheet: {missing}\n")
    sys.exit(1)

seen = set()
for row in rows[1:]:
    cells = row.findall('main:c', NS)
    values = []
    for c in cells:
        text = None
        t = c.find('.//main:t', NS)
        if t is not None:
            text = t.text
        v = c.find('main:v', NS)
        if text is None and v is not None:
            text = v.text
        values.append(text or '')
    record = dict(zip(headers, values))
    key = tuple(record.get(k, '') for k in required)
    if not all(key):
        continue
    if key not in seen:
        seen.add(key)
        print('\t'.join(key))
