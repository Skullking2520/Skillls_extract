import os
import json
import gspread
from gspread_dataframe import get_as_dataframe
from google.oauth2.service_account import Credentials
from skills_differ import skills as skill_list
from sentence_transformers import SentenceTransformer
import faiss, pandas as pd
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
    return client.open_by_key(os.environ["DRIVE_KEY"]).worksheet("Sheet1")

def main():
    model = SentenceTransformer("serbog/multilingual-e5-large-skill-job-matcher")
    skill_embs = model.encode(
        [str(s) for s in skill_list],
        convert_to_numpy=True,
        show_progress_bar=True
    )
    faiss.normalize_L2(skill_embs)
    index = faiss.IndexFlatIP(skill_embs.shape[1])
    index.add(skill_embs)

    ws = bring_sheet()
    df = get_as_dataframe(ws, evaluate_formulas=True)

    header = ws.row_values(1)
    if "skills" not in header:
        ws.update_cell(1, len(header) + 1, "skills")
        header.append("skills")
    skills_col = header.index("skills") + 1

    BATCH, buffer, cnt = 20, [], 0

    for r, row in enumerate(df.itertuples(index=False), start=2):
        if pd.notnull(getattr(row, "skills", None)):
            continue

        title = getattr(row, "job_title", "") or ""
        desc  = getattr(row, "description", "") or ""
        if not desc.strip():
            continue

        q_emb = model.encode([f"{title}. {desc}"], convert_to_numpy=True)
        faiss.normalize_L2(q_emb)
        D, I = index.search(q_emb, 20)
        matched = [
            skill_list[i]
            for i, score in zip(I[0], D[0])
            if score > 0.4
        ]
        skills_str = (
            ", ".join(f'"{s}"' for s in matched)
            if matched
            else "No skills found"
        )

        buffer.append(Cell(r, skills_col, skills_str))
        cnt += 1

        if cnt % BATCH == 0:
            ws.update_cells(buffer)
            buffer.clear()

    if buffer:
        ws.update_cells(buffer)

    print("Skills extraction completed.")

if __name__ == "__main__":
    main()
