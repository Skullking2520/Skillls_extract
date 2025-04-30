import os
import pandas as pd
import json
import gspread
from gspread_dataframe import get_as_dataframe
from google.oauth2.service_account import Credentials
from transformers import pipeline
from gspread import Cell
import numpy as np


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
    return client.open_by_key(os.environ["DRIVE_KEY"]).worksheet("Sheet1")

def main():
    ws = bring_sheet()
    df = get_as_dataframe(ws, evaluate_formulas=True)

    header = ws.row_values(1)
    if "skill_levels" not in header:
        ws.update_cell(1, len(header) + 1, "skill_levels")
        header.append("skill_levels")
    if "job_level" not in header:
        ws.update_cell(1, len(header) + 1, "job_level")
        header.append("job_level")

    skills_col = header.index("skills") + 1
    skill_lv_col = header.index("skill_levels") + 1
    job_lvl_col = header.index("job_level") + 1

    zs = pipeline(
        "zero-shot-classification",
        model="facebook/bart-large-mnli",
        hypothesis_template="This requires a {} level.",
        device=0
    )
    skill_level_labels = ["Novice", "Advanced Beginner", "Competent", "Proficient", "Expert"]

    level_to_num = {
        "Novice": 1,
        "Advanced Beginner": 2,
        "Competent": 3,
        "Proficient": 4,
        "Expert": 5
    }

    buffer, cnt, BATCH = [], 0, 20

    for r, row in enumerate(df.itertuples(index=False), start=2):
        if pd.notna(getattr(row, "skill_levels", None)) or not getattr(row, "skills", ""):
            continue

        title = getattr(row, "job_title", "") or ""
        desc = getattr(row, "description", "") or ""
        raw_skills = getattr(row, "skills", "")
        skills = [s.strip().strip('"') for s in raw_skills.split(",") if s.strip()]

        pairs = []
        nums = []
        text = f"{title}. {desc}"
        for sk in skills:
            res = zs(text, skill_level_labels, multi_label=False)
            lvl = res["labels"][0]
            pairs.append(f"{sk}={lvl}")
            nums.append(level_to_num[lvl])

        lv_str = ", ".join(pairs) if pairs else "None"
        avg = np.mean(nums) if nums else 0

        if avg == 0:
            jl = ""
        elif avg < 2:
            jl = "Entry Level"
        elif avg < 4:
            jl = "Mid Level"
        else:
            jl = "Senior Level"

        buffer += [
            Cell(r, skill_lv_col, lv_str),
            Cell(r, job_lvl_col, jl)
        ]
        cnt += 1

        if cnt % BATCH == 0:
            ws.update_cells(buffer)
            buffer.clear()

    if buffer:
        ws.update_cells(buffer)


if __name__ == "__main__":
    main()
