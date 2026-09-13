"""Reshape src/experiment.py's JSON output into the row-per-condition CSV
format compare_to_paper.py expects (Model Name,Condition,f1,precision,
easy_f1,medium_f1,hard_f1), plus knowledge/logic slice columns kept alongside
for the RAG-vs-CoT taxonomy check.
"""
import json
import sys
import csv

def rows_for(json_path, model_name):
    d = json.load(open(json_path))
    rows = []
    for cond, report in d["results"].items():
        o = report["overall"]
        rows.append({
            "Model Name": model_name,
            "Condition": cond,
            "f1": o["f1"],
            "precision": o["precision"],
            "recall": o["recall"],
            "easy_f1": report["easy"]["f1"],
            "medium_f1": report["medium"]["f1"],
            "hard_f1": report["hard"]["f1"],
            "knowledge_f1": (report.get("type:knowledge") or {}).get("f1", ""),
            "logic_f1": (report.get("type:logic") or {}).get("f1", ""),
            "n_rows": d["rows"],
            "corpus": d.get("rag", {}).get("corpus", ""),
        })
    return rows

if __name__ == "__main__":
    out_path = sys.argv[1]
    pairs = sys.argv[2:]  # json_path:model_name pairs
    all_rows = []
    for pair in pairs:
        jp, name = pair.split(":", 1)
        all_rows.extend(rows_for(jp, name))
    fieldnames = ["Model Name", "Condition", "f1", "precision", "recall",
                  "easy_f1", "medium_f1", "hard_f1", "knowledge_f1", "logic_f1",
                  "n_rows", "corpus"]
    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(all_rows)
    print(f"wrote {len(all_rows)} rows -> {out_path}")
