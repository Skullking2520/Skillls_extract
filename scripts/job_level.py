import os
import json
import pandas as pd
import gspread
from gspread_dataframe import get_as_dataframe
from google.oauth2.service_account import Credentials
from transformers import pipeline
from gspread import Cell

def bring_sheet():
    creds_info = json.loads(os.environ["GCP_SA_KEY"])
    creds = Credentials.from_service_account_info(
        creds_info,
        scopes=[
            "https://www.googleapis.com/auth/spreadsheets",
            "https://www.googleapis.com/auth/drive"
        ]
    )
    client = gspread.authorize(creds)
    return client.open_by_key(os.environ["DRIVE_KEY"]).worksheet("Skills Extraction")

def main():
    ws = bring_sheet()
    df = get_as_dataframe(ws, evaluate_formulas=True)

    header = ws.row_values(1)
    if "job_level" not in header:
        ws.update_cell(1, len(header) + 1, "job_level")
        header.append("job_level")
    job_lvl_col = header.index("job_level") + 1

    zs = pipeline(
        "zero-shot-classification",
        model="facebook/bart-large-mnli",
        hypothesis_template="This position is {}.",
        device=-1
    )
    candidate_labels = [
        "Executive or senior level",
        "Middle level",
        "First-level",
        "Intermediate or experienced (senior staff) level",
        "Entry-level"
    ]

    BATCH, buffer, cnt = 20, [], 0
    for r, row in enumerate(df.itertuples(index=False), start=2):
        if pd.notna(getattr(row, "job_level", None)):
            continue

        title = getattr(row, "job_title", "") or ""
        desc  = getattr(row, "description", "") or ""
        text  = f"{title}. {desc}".strip()
        if not text:
            continue

        res = zs(text, candidate_labels, multi_label=False)
        lvl = res["labels"][0]

        buffer.append(Cell(r, job_lvl_col, lvl))
        cnt += 1

        if cnt % BATCH == 0:
            ws.update_cells(buffer)
            buffer.clear()

    if buffer:
        ws.update_cells(buffer)

    print("Job level classification completed.")

if __name__ == "__main__":
    main()
