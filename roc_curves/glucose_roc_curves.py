"""
Univariate ROC: for every pairwise comparison of the 4 study groups, how well
does serum glucose alone rank one group above the other? Uses every
participant (train+test combined) since nothing is being fit.

Output: glucose_roc_curves.png
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ablation_common import GLUCOSE_COL, load_data
from roc_common import make_pairwise_roc_plot

OUT_DIR = os.path.dirname(os.path.abspath(__file__))

df = load_data()
df = df.dropna(subset=[GLUCOSE_COL])

make_pairwise_roc_plot(
    df, GLUCOSE_COL,
    "Serum glucose as a univariate classifier: all pairwise study-group comparisons",
    os.path.join(OUT_DIR, "glucose_roc_curves.png"),
)
