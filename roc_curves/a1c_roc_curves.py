"""
Univariate ROC: for every pairwise comparison of the 4 study groups, how well
does HbA1c alone rank one group above the other? Uses every participant
(train+test combined) since nothing is being fit.

Output: a1c_roc_curves.png
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ablation_common import HBA1C_COL, load_data
from roc_common import make_pairwise_roc_plot

OUT_DIR = os.path.dirname(os.path.abspath(__file__))

df = load_data()
df = df.dropna(subset=[HBA1C_COL])

make_pairwise_roc_plot(
    df, HBA1C_COL,
    "HbA1c as a univariate classifier: all pairwise study-group comparisons",
    os.path.join(OUT_DIR, "a1c_roc_curves.png"),
)
