import os
import pandas as pd
import requests

BASE_URL = "https://s3.amazonaws.com/fcp-indi/data/Projects/ABIDE_Initiative/Outputs"
PIPELINE = "cpac"
STRATEGY = "filt_global"
DERIVATIVE = "rois_cc200"

phenos = pd.read_csv("../data/phenotypes/Phenotypic_V1_0b_preprocessed1.csv")

out_dir = "../data/functionals/cpac/filt_global/rois_cc200"
os.makedirs(out_dir, exist_ok=True)

missing = []

for fid in phenos["FILE_ID"]:
    url = f"{BASE_URL}/{PIPELINE}/{STRATEGY}/{DERIVATIVE}/{fid}_{DERIVATIVE}.1D"
    out_path = os.path.join(out_dir, f"{fid}_{DERIVATIVE}.1D")

    if os.path.exists(out_path):
        continue

    r = requests.get(url)
    if r.status_code == 200:
        with open(out_path, "wb") as f:
            f.write(r.content)
        print("Downloaded:", fid)
    else:
        print("Missing:", fid)
        missing.append(fid)

print("Finished. Missing files:", len(missing))
