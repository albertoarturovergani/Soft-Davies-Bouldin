from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'/'python'))
from summarize_abstract import load_scores,aggregate,report,two_way_anova

ROOT=Path(__file__).resolve().parents[1]
DIR=ROOT/'results'/'abstract_reproduction'
FILES=[DIR/'ground_100seeds_k2-10_scores.csv.gz',DIR/'fmri_10seeds_k2-10_scores.csv.gz']

def test_2018_abstract_counts_from_recomputed_scores():
    df=load_scores(DIR,FILES,100,10)
    assert len(df)==13320
    avg=aggregate(df,ground_runs=100,fmri_runs=10)
    aim,rim,mrc=report(avg)
    assert int(aim.SDB_below_both.sum())==7
    assert int(rim.SDB_below_both.sum())==4
    assert int(mrc.SDB_below_both.sum())==9
    true_k={'Cancer':2,'Crab':2,'Glass':2,'Iris':3,'Ovarian':2,'Simple':4,'Thyroid':3,'Wine':3}
    assert int(sum(row['best_k_SDB_cmeans']==true_k.get(row['dataset'],-1) for row in aim.to_dict('records')))==3
    assert two_way_anova(aim,'dataset')['p_method']>0.05
    assert two_way_anova(rim,'dataset')['p_method']>0.05
