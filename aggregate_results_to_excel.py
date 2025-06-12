# Script to aggregate statistical test results from unit_model/statistics/results
# into an Excel (.xlsx) file. Produces summary.csv (CSV, compatible with Excel)
# and summary.xlsx (minimal xlsx written without external libraries).

import json
import os
import csv
import zipfile

# Directory containing this script
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# Location of statistical result folders
RESULTS_DIR = os.path.join(BASE_DIR, 'unit_model', 'statistics', 'results')

# Known model type strings
KNOWN_MODELS = [
    'GAT', 'GCN', 'GIN', 'GCN_GAT', 'GIN_GAT', 'GIN_GCN', 'GIN_GCN_GAT'
]

def parse_target(target: str):
    """Return (dataset, model) from target string."""
    parts = target.split('_')
    for i in range(3, 0, -1):
        if len(parts) >= i:
            m = '_'.join(parts[-i:])
            if m in KNOWN_MODELS:
                dataset = '_'.join(parts[:-i])
                return dataset, m
    # fallback: last element is model
    return '_'.join(parts[:-1]), parts[-1]

def collect_results():
    rows = []
    for folder in os.listdir(RESULTS_DIR):
        path = os.path.join(RESULTS_DIR, folder)
        if not os.path.isdir(path):
            continue
        src, tgt = folder.split('&&', 1)
        tgt_dataset, model = parse_target(tgt)
        stat_path = os.path.join(path, 'stat_test.json')
        try:
            with open(stat_path) as f:
                data = json.load(f)
        except FileNotFoundError:
            continue
        row = {
            'source': src,
            'target': tgt_dataset,
            'model': model,
            't_stat': data.get('t_test', {}).get('t_stat'),
            't_p_value': data.get('t_test', {}).get('p_value'),
            'w_stat': data.get('wilcoxon', {}).get('w_stat'),
            'w_p_value': data.get('wilcoxon', {}).get('p_value'),
        }
        rows.append(row)
    return rows

def write_csv(rows, path):
    headers = ['source', 'target', 'model', 't_stat', 't_p_value', 'w_stat', 'w_p_value']
    with open(path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        writer.writerows(rows)

# Minimal Excel writer using only the standard library
# Creates a very simple XLSX with one worksheet

def write_xlsx(rows, path):
    headers = ['source', 'target', 'model', 't_stat', 't_p_value', 'w_stat', 'w_p_value']
    # Build sheet XML
    def escape(s):
        return (s.replace('&', '&amp;').replace('<', '&lt;')
                .replace('>', '&gt;').replace('"', '&quot;'))

    shared_strings = []
    def shared_index(value):
        if value not in shared_strings:
            shared_strings.append(value)
        return shared_strings.index(value)

    rows_xml = []
    # header row
    cells = []
    for col, h in enumerate(headers, 1):
        si = shared_index(h)
        cells.append(f'<c r="{chr(64+col)}1" t="s"><v>{si}</v></c>')
    rows_xml.append(f'<row r="1">{"".join(cells)}</row>')
    # data rows
    for r, row in enumerate(rows, start=2):
        cells = []
        for c, h in enumerate(headers, 1):
            val = row[h]
            if isinstance(val, (int, float)) and val is not None:
                cells.append(f'<c r="{chr(64+c)}{r}"><v>{val}</v></c>')
            else:
                text = '' if val is None else str(val)
                si = shared_index(text)
                cells.append(f'<c r="{chr(64+c)}{r}" t="s"><v>{si}</v></c>')
        rows_xml.append(f'<row r="{r}">{"".join(cells)}</row>')
    sheet_xml = (
        '<?xml version="1.0" encoding="UTF-8"?>'
        '<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">'
        '<sheetData>' + ''.join(rows_xml) + '</sheetData>'
        '</worksheet>'
    )

    shared_xml = (
        '<?xml version="1.0" encoding="UTF-8"?>'
        '<sst xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" count="{0}" uniqueCount="{0}">'
        .format(len(shared_strings)) + ''.join(
            f'<si><t>{escape(s)}</t></si>' for s in shared_strings
        ) + '</sst>'
    )

    workbook_xml = (
        '<?xml version="1.0" encoding="UTF-8"?>'
        '<workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main"'
        ' xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">'
        '<sheets><sheet name="Sheet1" sheetId="1" r:id="rId1"/></sheets></workbook>'
    )

    rels_xml = (
        '<?xml version="1.0" encoding="UTF-8"?>'
        '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
        '<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" Target="worksheets/sheet1.xml"/>'
        '<Relationship Id="rId2" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/sharedStrings" Target="sharedStrings.xml"/>'
        '</Relationships>'
    )

    content_types = (
        '<?xml version="1.0" encoding="UTF-8"?>'
        '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
        '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>'
        '<Default Extension="xml" ContentType="application/xml"/>'
        '<Override PartName="/xl/worksheets/sheet1.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/>'
        '<Override PartName="/xl/workbook.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml"/>'
        '<Override PartName="/xl/sharedStrings.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sharedStrings+xml"/>'
        '</Types>'
    )

    wb_rels = (
        '<?xml version="1.0" encoding="UTF-8"?>'
        '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
        '<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" Target="worksheets/sheet1.xml"/>'
        '<Relationship Id="rId2" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/sharedStrings" Target="sharedStrings.xml"/>'
        '</Relationships>'
    )

    with zipfile.ZipFile(path, 'w') as z:
        z.writestr('[Content_Types].xml', content_types)
        z.writestr('_rels/.rels', (
            '<?xml version="1.0" encoding="UTF-8"?>'
            '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
            '<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="xl/workbook.xml"/>'
            '</Relationships>'
        ))
        z.writestr('xl/workbook.xml', workbook_xml)
        z.writestr('xl/worksheets/sheet1.xml', sheet_xml)
        z.writestr('xl/sharedStrings.xml', shared_xml)
        z.writestr('xl/_rels/workbook.xml.rels', rels_xml)

if __name__ == '__main__':
    rows = collect_results()
    rows.sort(key=lambda r: (r['source'], r['target'], r['model']))
    write_csv(rows, 'summary.csv')
    try:
        write_xlsx(rows, 'summary.xlsx')
        print('Wrote summary.xlsx with', len(rows), 'rows')
    except Exception as e:
        print('Failed to write xlsx:', e)
        print('CSV written as summary.csv')
