"""
Univariate check: does raw serum insulin alone separate participants across
the study groups (no model, no other features)?

Why this is separate from run_classification_ablation.py: the ablation study
asks whether insulin as a *predictor variable* adds value on top of everything
else (and in fact never uses it, since it's excluded as a leakage-adjacent
feature). This script asks a narrower, prior question -- given that insulin is
reportedly significantly different between groups (e.g. a t-test/boxplot
comparison), is that difference actually large enough for insulin BY ITSELF to
rank participants correctly? A significant group difference in means only says
the distributions aren't identical; the AUROC below says how well a single
insulin value would let you rank a random participant from one group above a
random participant from another, for every pairwise group comparison. Uses
every participant (train+test combined) since nothing is being fit -- there's
no held-out set to protect.

Output: insulin_univariate_roc.png
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ablation_common import INSULIN_COL, load_data

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "roc_curves"))
from roc_common import make_pairwise_roc_plot

OUT_DIR = os.path.dirname(os.path.abspath(__file__))

df = load_data()
df = df.dropna(subset=[INSULIN_COL])

make_pairwise_roc_plot(
    df, INSULIN_COL,
    "Serum insulin as a univariate classifier: all pairwise study-group comparisons",
    os.path.join(OUT_DIR, "insulin_univariate_roc.png"),
)
