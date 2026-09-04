import pandas as pd
from pathlib import Path
import argparse

def main():
    """
    Take in raw chat log. It is stored in XLS file, but is actually HTML format.
    Anonymize by dropping PII - Personally Identifiable Information
    Expects following folder structure:
    1- Raw chat logs in a folder outside this data prep folder
    2- A folder to store anonymized data outside this data prep folder
    File format expectations:
    1- Raw chat logs as HTML/XML stored in an XLS file
    2- Anonymized chat logs will be stored as CSV: structure is column name transcript and one line per full student conversation
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("filename")
    parser.add_argument("--guardrail", action="store_true")
    args = parser.parse_args()

    basePath = Path("..") / args.filename
    finalDirectory = Path("../anonymized_data")

    if args.guardrail: finalPath = finalDirectory / f"{Path(args.filename).stem}_GR.csv"
    else: finalPath = finalDirectory / f"{Path(args.filename).stem}.csv"

    tables = pd.read_html(basePath)

    df = tables[0]
    df = df[["Transcript"]].dropna().reset_index(drop=True)
    df.to_csv(finalPath, index=False)

    print(f"Anonymized data saved to: {finalPath}")

if __name__=="__main__":
    main()
