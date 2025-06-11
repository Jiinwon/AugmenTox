import pandas as pd
from scipy.stats import ttest_rel
import os

EXCEL_PATH = os.path.join(os.path.dirname(__file__), '..', 'performance_summary.xlsx')

# Load the DataFrame from the 'p-value' sheet

df = pd.read_excel(EXCEL_PATH, sheet_name='p-value')

required = {'source', 'target', 'model_type', 'F1_finetune', 'F1_targetonly'}
missing = required - set(df.columns)
if missing:
    raise ValueError(f"Missing columns: {missing}")

results = []

for (src, tgt, model), g in df.groupby(['source', 'target', 'model_type']):
    sub = g[['F1_finetune', 'F1_targetonly']].dropna()
    if len(sub) < 2:
        continue
    t_stat, p_val = ttest_rel(sub['F1_finetune'], sub['F1_targetonly'])
    results.append({
        'source': src,
        'target': tgt,
        'model_type': model,
        'n': len(sub),
        't_stat': t_stat,
        'p_value': p_val
    })

out_df = pd.DataFrame(results)

# Save results back into the Excel file
with pd.ExcelWriter(EXCEL_PATH, mode='a', if_sheet_exists='replace') as writer:
    out_df.to_excel(writer, sheet_name='p-value-results', index=False)

print('✅ p-value calculation complete')
