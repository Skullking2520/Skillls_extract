import os
import pandas as pd
import json
import gspread
from gspread_dataframe import get_as_dataframe
from google.oauth2.service_account import Credentials
from transformers import pipeline

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
    if "skill_levels" not in header:
        ws.update_cell(1, len(header) + 1, "skill_levels")
        header.append("skill_levels")
    skill_lv_col = header.index("skill_levels") + 1

    zs = pipeline(
        "zero-shot-classification",
        model="facebook/bart-large-mnli",
        device=-1
    )
    skill_level_labels = ["Novice", "Advanced Beginner", "Competent", "Proficient", "Expert"]

    for r, row in enumerate(df.itertuples(index=False), start=2):
        raw = getattr(row, "skills", None)
        if pd.isna(raw) or not str(raw).strip():
            continue
        if pd.notna(getattr(row, "skill_levels", None)):
            continue

        title = getattr(row, "job_title", "") or ""
        desc = getattr(row, "description", "") or ""
        skills = [s.strip().strip('"') for s in str(raw).split(",") if s.strip()]

        pairs = []
        base_text = f"{title}. {desc}"
        for sk in skills:
            text = f"{base_text} Skill: {sk}."
            res = zs(
                text,
                skill_level_labels,
                hypothesis_template=f"The job requires {{}} proficiency in {sk}.",
                multi_label=False
            )
            lvl = res["labels"][0]
            pairs.append(f"{sk}={lvl}")

        lv_str = ", ".join(pairs) if pairs else "None"
        ws.update_cell(r, skill_lv_col, lv_str)

    print("skill levels updated")

if __name__ == "__main__":
    main()
