import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats
from scipy.special import gammaln, logsumexp
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
import joblib

CATEGORIES = {
    'sex':['Male','Female'],
    'smoking':['Never','Previous','Current','Unknown'],
    'alcohol':['Current','Never','Previous','Unknown'],
    'diabetes':['No','Yes','Unknown'],
    'vascular':['None','Any','Unknown']}


def bh_adjust(pvalues):
    p=np.asarray(pvalues,dtype=float)
    if p.ndim!=1 or not np.isfinite(p).all() or ((p<0)|(p>1)).any():
        raise ValueError('')
    order=np.argsort(p);q=np.minimum.accumulate((p[order]*len(p)/np.arange(1,len(p)+1))[::-1])[::-1]
    out=np.empty_like(p);out[order]=np.minimum(q,1)
    return out



def fisher_exact_2xk(table, max_tables=10000000):
    raw=np.asarray(table)
    if raw.ndim!=2 or raw.shape[0]!=2 or raw.shape[1]<2 or not np.isfinite(raw).all():
        raise ValueError('')
    if (raw<0).any() or not np.equal(raw,np.floor(raw)).all():
        raise ValueError('Counts must be non-negative integers.')
    table=raw.astype(np.int64)
    table=table[:,table.sum(axis=0)>0]
    if table.shape[1]<2 or (table.sum(axis=1)==0).any():
        raise ValueError('')
    table=table[:,np.argsort(table.sum(axis=0))]
    cols=table.sum(axis=0);total=int(cols.sum());row=int(table[0].sum())
    logchoose=lambda n,k: gammaln(n+1)-gammaln(k+1)-gammaln(n-k+1)
    normalizer=float(logchoose(total,row))
    observed=float(logchoose(cols,table[0]).sum())-normalizer
    remaining=np.cumsum(cols[::-1])[::-1]
    lookups=[logchoose(int(n),np.arange(int(n)+1)) for n in cols]
    logp=-np.inf;visited=0
    def visit(j,left,partial):
        nonlocal logp,visited
        if j==len(cols)-2:
            low=max(0,left-int(cols[j+1]));high=min(int(cols[j]),left)
            if high<low:return
            xs=np.arange(low,high+1,dtype=np.int64)
            visited+=len(xs)
            if visited>int(max_tables):
                raise RuntimeError('')
            values=partial+lookups[j][xs]+lookups[j+1][left-xs]-normalizer
            selected=values[values<=observed+1e-10]
            if len(selected):logp=np.logaddexp(logp,logsumexp(selected))
            return
        low=max(0,left-int(remaining[j+1]));high=min(int(cols[j]),left)
        for x in range(low,high+1):visit(j+1,left-x,partial+lookups[j][x])
    visit(0,row,0.)
    return {'pvalue':min(1.,float(np.exp(logp))),'tables_enumerated':visited,
            'definition':'fixed margins'}


def normalize_covariates(df):
    df=df.copy()
    for name in ['age','bmi']:
        if name in df:
            df[name]=pd.to_numeric(df[name],errors='raise')
            if (df[name].dropna()<=0).any():
                raise ValueError('')
    vascular=df.vascular.astype('string').str.strip().replace({'0':'None','0.0':'None','1':'Any','1.0':'Any'})
    df['vascular']=vascular.fillna('Unknown')
    for name,allowed in CATEGORIES.items():
        if name not in df:
            continue
        df[name]=df[name].astype('string').str.strip().fillna('Unknown')
        invalid=~df[name].isin(allowed + (['Unknown'] if name=='sex' else []))
        if invalid.any():
            raise ValueError(f'{name}: unrecognized categories {df.loc[invalid,name].unique().tolist()}; allowed {allowed}.')
    return df


def build_baseline(c):
    scaler=ColumnTransformer([('age',StandardScaler(),['age'])],remainder='passthrough',verbose_feature_names_out=False)
    return Pipeline([('scaler',scaler),('clf',LogisticRegression(C=float(c['baseline']['C']),l1_ratio=0,solver='lbfgs',max_iter=1000))])


def baseline_analysis(df,c,out):
    data=df.copy()
    data['vascular_binary']=data.vascular.map({'None':0.,'Any':1.})
    complete=data.age.notna() & data.vascular_binary.notna()
    data.loc[~complete,['participant_id','split','age','vascular']].to_csv(out/'baseline_excluded.csv',index=False)
    data=data.loc[complete].copy()
    X=data[['age','vascular_binary']];y=data.label
    train=data.split.isin(c['baseline']['fit_splits']);test=data.split=='test'
    if set(y[train])!={0,1} or set(y[test])!={0,1}:
        raise ValueError('Baseline needs both classes in complete-case development and test data.')
    model=build_baseline(c).fit(X.loc[train],y.loc[train])
    scores=model.predict_proba(X.loc[test])[:,1]
    result=data.loc[test,['participant_id','label','split']].copy();result['p_crc']=scores
    result.to_csv(out/'baseline_test_predictions.csv',index=False)
    plot_roc(y[test],scores,'Age + vascular/heart disease',out/'baseline_roc.png')
    test_auc=float(roc_auc_score(y[test],scores))
    k=int(c['baseline']['folds'])
    if y.value_counts().min()<k:
        raise ValueError(f'Need at least {k} complete cases in each class for CV.')
    skf=StratifiedKFold(n_splits=k,shuffle=True,random_state=int(c['seed']))
    aucs=[];oof=[]
    for fold,(tr,va) in enumerate(skf.split(X,y),1):
        fitted=build_baseline(c).fit(X.iloc[tr],y.iloc[tr])
        p=fitted.predict_proba(X.iloc[va])[:,1]
        aucs.append(float(roc_auc_score(y.iloc[va],p)))
        rows=data.iloc[va][['participant_id','label']].copy();rows['fold']=fold;rows['p_crc']=p;oof.append(rows)
    pd.concat(oof).to_csv(out/'baseline_cv_predictions.csv',index=False)
    scale=model.named_steps['scaler'].named_transformers_['age']
    coef=model.named_steps['clf']
    audit={'n_input':len(df),'n_complete':len(data),'n_excluded':len(df)-len(data),
           'counts_by_split_and_label':data.groupby(['split','label']).size().unstack(fill_value=0).to_dict(),
           'fit_splits':c['baseline']['fit_splits'],'test_source':'same participant IDs as imaging manifest, complete cases only',
           'held_out_test_auroc':test_auc,'test_ci95':hanley_mcneil(test_auc,int(y[test].sum()),int((y[test]==0).sum())),
           'cv_mean_auroc':float(np.mean(aucs)),'cv_std_auroc':float(np.std(aucs,ddof=c['baseline']['cv_sd_ddof'])),
           'cv_sd_ddof':c['baseline']['cv_sd_ddof'],'cv_folds':aucs,'cv_population':'',
           'scaler_age_mean':float(scale.mean_[0]),'scaler_age_scale':float(scale.scale_[0]),
           'coefficients':{'age_z':float(coef.coef_[0,0]),'vascular_0_1':float(coef.coef_[0,1]),'intercept':float(coef.intercept_[0])}}
    save_json(audit,out/'baseline_metrics.json')
    joblib.dump(model,out/'baseline_model.joblib')
    return audit


def descriptive_statistics(df,c,out):
    family=c['statistics']['fdr_family']
    expected=['age','sex','bmi','smoking','alcohol','diabetes','vascular']
    if set(family)!=set(expected) or len(family)!=len(expected):
        raise ValueError('')
    missing=set(expected)-set(df)
    if missing:
        raise ValueError('')
    rows=[];details={};normality={}
    for name in family:
        if name in ['age','bmi']:
            a=df.loc[df.label==0,name].dropna().to_numpy(float)
            b=df.loc[df.label==1,name].dropna().to_numpy(float)
            if min(len(a),len(b))<3:
                raise ValueError('')
            u=stats.mannwhitneyu(a,b,alternative='two-sided',method='asymptotic')
            normality[name]={str(k):{'W':float(stats.shapiro(v).statistic),'p':float(stats.shapiro(v).pvalue)} for k,v in [(0,a),(1,b)]}
            effect=2*u.statistic/(len(a)*len(b))-1
            rows.append({'variable':name,'test':'Mann-Whitney U, asymptotic, two-sided', 'p':float(u.pvalue),
                         'n_non_crc':len(a),'n_crc':len(b),'effect_type':'Cliff delta: non-CRC minus CRC','effect':effect,
                         'non_crc_mean':float(a.mean()),'non_crc_sd':float(a.std(ddof=1)),
                         'crc_mean':float(b.mean()),'crc_sd':float(b.std(ddof=1))})
        else:
            allowed=CATEGORIES[name]+(['Unknown'] if name=='sex' else [])
            tab=pd.crosstab(df.label,df[name]).reindex(index=[0,1],columns=allowed,fill_value=0)
            tab=tab.loc[:,tab.sum(axis=0)>0]
            if tab.shape[1]<2:
                raise ValueError('')
            chi,p,dof,exp=stats.chi2_contingency(tab.values,correction=False)
            chosen='Pearson chi-square, no Yates correction';exact_details=None
            if (exp<5).any():
                if tab.shape==(2,2):
                    p=float(stats.fisher_exact(tab.values,alternative='two-sided').pvalue);chosen='Fisher exact, two-sided'
                else:
                    exact_details=fisher_exact_2xk(tab.values,c['statistics'].get('fisher_exact_max_tables',10000000))
                    p=exact_details['pvalue'];chosen='Fisher-Freeman-Halton exact, two-sided'
            v=float(np.sqrt(chi/(tab.values.sum()*min(tab.shape[0]-1,tab.shape[1]-1))))
            effect_type='Cramer V (uncorrected Pearson statistic)';effect=v
            if name=='sex' and set(tab.columns)=={'Male','Female'}:
                a=float(tab.loc[1,'Male']);b=float(tab.loc[1,'Female']);cc=float(tab.loc[0,'Male']);d=float(tab.loc[0,'Female'])
                effect=a*d/(b*cc) if b*cc else None
                effect_type=''
            details[name]={'categories':tab.columns.tolist(),'counts_non_crc':tab.loc[0].tolist(),'counts_crc':tab.loc[1].tolist(),
                           'percent_non_crc':(100*tab.loc[0]/tab.loc[0].sum()).tolist(),'percent_crc':(100*tab.loc[1]/tab.loc[1].sum()).tolist(),
                           'expected':exp.tolist(),'pearson_chi2':float(chi),'df':int(dof),'cramer_v':v,'exact_fisher':exact_details}
            rows.append({'variable':name,'test':chosen,'p':float(p),'n_non_crc':int(tab.loc[0].sum()),'n_crc':int(tab.loc[1].sum()),
                         'effect_type':effect_type,'effect':effect})
    q=bh_adjust([r['p'] for r in rows])
    for r,adjusted in zip(rows,q):
        r['fdr_q']=float(adjusted);r['significant_fdr_005']=bool(adjusted<.05)
    pd.DataFrame(rows).to_csv(out/'table4_statistics.csv',index=False)
    category_rows=[]
    for variable,d in details.items():
        for i,category in enumerate(d['categories']):
            category_rows.append({'variable':variable,'category':category,
                'non_crc_n':d['counts_non_crc'][i],'non_crc_pct':d['percent_non_crc'][i],
                'crc_n':d['counts_crc'][i],'crc_pct':d['percent_crc'][i]})
    pd.DataFrame(category_rows).to_csv(out/'table4_categories.csv',index=False)
    return rows


def main_baseline():
    ap=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='')
    ap.add_argument('--config',default='config.json');ap.add_argument('--csv',required=True)
    ap.add_argument('--manifest',required=True)
    ap.add_argument('--out',default='baseline_out')
    ap.add_argument('--baseline-only',action='store_true',)
    ap.add_argument('--stats-only',action='store_true')
    a=ap.parse_args()
    if a.baseline_only and a.stats_only:
        ap.error('')
    c=load_config(a.config);manifest=read_manifest(a.manifest);assert_split_ready(manifest)
    df=normalize_covariates(load_covariates(a.csv,manifest));out=new_output(a.out)
   
    if not a.stats_only:
        result=baseline_analysis(df,c,out)
        print(f'Baseline test AUROC: {result["held_out_test_auroc"]:.4f}')
    if not a.baseline_only:
        descriptive_statistics(df,c,out)
    print(f'Saved: {out}')

from train_crc_b3 import (
    load_config, read_manifest, assert_split_ready, load_covariates, new_output,
    save_json, sha256, environment, hanley_mcneil, plot_roc,
    verify_images, load_checkpoint, build_transforms, autocast, seed_everything
)

if __name__ == "__main__":
    main_baseline()
