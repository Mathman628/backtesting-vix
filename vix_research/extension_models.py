"""Causal, fixed-three-state adaptation of arXiv:2603.04441, not a replication."""
import numpy as np
import pandas as pd
from scipy.special import logsumexp
from scipy.optimize import minimize, linear_sum_assignment
from hmmlearn.hmm import GaussianHMM
from sklearn.covariance import LedoitWolf
from threadpoolctl import threadpool_limits
from .core import ASSETS, Decision, cap_portfolio_vol, estimated_vol


def filter_step(prior, observation, means, variances):
    logp=-.5*np.sum(np.log(2*np.pi*variances)+(observation-means)**2/variances,axis=1)
    score=np.log(np.maximum(prior,1e-300))+logp
    return np.exp(score-logsumexp(score))


def mixture(p, means, covariances):
    mu=p@means
    delta=means-mu
    cov=np.einsum('k,kij->ij',p,covariances)+np.einsum('k,ki,kj->ij',p,delta,delta)
    return mu,cov


def wasserstein_cost(means, variances, anchor_means, anchor_variances):
    return ((means[:,None]-anchor_means[None,:])**2+
            (np.sqrt(variances[:,None])-np.sqrt(anchor_variances[None,:]))**2).sum(axis=2)


def forecasts(closes, start='2010-01-04', window=756):
    """Monthly fits on past data only; daily forward filter, next-state forecasts.

    Templates use fixed initial units and one-to-one Wasserstein alignment.
    Return-moment EMA is an explicit stabilisation adaptation. No state decoding
    or future smoothed probabilities enter the evaluation period.
    """
    logret=np.log(closes[list(ASSETS[:3])]).diff()
    features=pd.concat([logret,logret.rolling(60).std(),logret.rolling(20).mean()],axis=1)
    simple=closes[list(ASSETS)].pct_change(fill_method=None)
    valid=features.notna().all(axis=1)&simple.notna().all(axis=1)
    f=features.loc[valid];r=simple.loc[valid]
    output={}; diagnostics=[];month=None;reference=None
    with threadpool_limits(limits=1):
        for day in closes.loc[start:].index:
            history=f.loc[f.index<day]
            if len(history)<window:raise ValueError('Insufficient HMM training history')
            stamp=history.index[-1];key=(day.year,day.month)
            if key!=month:
                h=history.iloc[-window:];x=h.to_numpy();y=r.loc[h.index].to_numpy()
                center=x.mean(axis=0);scale=np.maximum(x.std(axis=0),1e-8);z=(x-center)/scale
                model=GaussianHMM(n_components=3,covariance_type='diag',n_iter=100,tol=.01,min_covar=.001,random_state=1729)
                model.fit(z)
                if not np.isfinite(model.score(z)):raise ArithmeticError('Invalid HMM fit')
                means=model.means_;variances=np.diagonal(model.covars_,axis1=1,axis2=2).copy()
                # Historical smoothing is used only within the already observed fit window.
                responsibility=model.predict_proba(z)
                global_mu=y.mean(axis=0);global_cov=LedoitWolf().fit(y).covariance_+np.eye(4)*1e-10
                state_mu=[];state_cov=[]
                for k in range(3):
                    w=responsibility[:,k];w=w/w.sum();m=w@y;d=y-m
                    state_mu.append(.5*m+.5*global_mu)
                    state_cov.append(.75*(d*w[:,None]).T@d+.25*global_cov)
                state_mu=np.array(state_mu);state_cov=np.array(state_cov)
                posterior=model.startprob_
                for obs in z:
                    posterior=filter_step(posterior,obs,means,variances)
                    next_prior=posterior@model.transmat_
                    posterior=next_prior
                # posterior now predicts the next session after the last fitted observation.
                if reference is None:reference=scale.copy()
                raw_mean=(means*scale+center)/reference;raw_var=variances*(scale/reference)**2
                if month is None:
                    template_mean=raw_mean.copy();template_var=raw_var.copy()
                    template_mu=state_mu.copy();template_cov=state_cov.copy();mapping=np.arange(3)
                else:
                    rows,cols=linear_sum_assignment(wasserstein_cost(raw_mean,raw_var,template_mean,template_var))
                    mapping=np.empty(3,dtype=int);mapping[rows]=cols
                    for k,j in enumerate(mapping):
                        template_mean[j]=.8*template_mean[j]+.2*raw_mean[k]
                        template_var[j]=.8*template_var[j]+.2*raw_var[k]
                        template_mu[j]=.8*template_mu[j]+.2*state_mu[k]
                        template_cov[j]=.8*template_cov[j]+.2*state_cov[k]
                diagnostics.append(dict(execution_date=day,train_start=h.index[0],train_end=stamp,observations=len(h),iterations=model.monitor_.iter,hit_iteration_limit=model.monitor_.iter>=100,log_likelihood=model.score(z)))
                month=key
            else:
                obs=(history.iloc[-1].to_numpy()-center)/scale
                posterior=filter_step(posterior,obs,means,variances)@model.transmat_
            stable_p=np.zeros(3);stable_p[mapping]=posterior
            output[day]={'unconditional':(global_mu,global_cov),'hmm':mixture(posterior,state_mu,state_cov),
                         'tracked':mixture(stable_p,template_mu,template_cov),'probabilities':stable_p.copy(),'signal_date':stamp}
            if day.month==1 and day.day<8:print('Causal forecasts:',day.date(),flush=True)
    return output,pd.DataFrame(diagnostics)


def optimise(mu,cov,current,penalty=.0006):
    # Auxiliary variables implement L1 transaction penalty without abs derivatives.
    def objective(x):return 1e4*(5*(-mu@x[:4]+5*x[:4]@cov@x[:4])+penalty*x[4:].sum())
    def gradient(x):return 1e4*np.r_[5*(-mu+10*cov@x[:4]),np.full(4,penalty)]
    constraints=[{'type':'eq','fun':lambda x:x[:4].sum()-1},
                 {'type':'ineq','fun':lambda x:x[4:]-(x[:4]-current)},
                 {'type':'ineq','fun':lambda x:x[4:]+(x[:4]-current)}]
    initial=np.array([0.,0.,0.,1.]);x0=np.r_[initial,abs(initial-current)]
    fit=minimize(objective,x0,jac=gradient,method='SLSQP',bounds=[(0,.6)]*3+[(0,1)]+[(0,2)]*4,
                 constraints=constraints,options={'ftol':1e-9,'maxiter':150})
    if not fit.success or abs(fit.x[:4].sum()-1)>1e-6:raise ArithmeticError('Optimiser failed: '+fit.message)
    return fit.x[:4]


class OptimisedPolicy:
    def __init__(self,cache,mode='hmm',penalty=.0006):
        self.cache=cache;self.mode=mode;self.penalty=penalty;self.week=None
    def decide(self,snapshot,execution_date,actual_weights):
        day=pd.Timestamp(execution_date);week=tuple(day.isocalendar()[:2]);weekly=week!=self.week;self.week=week
        cv=estimated_vol(actual_weights,snapshot.covariance)
        if not weekly and cv<=.12:return Decision(snapshot.date,day.date(),-1,actual_weights,False,'hold',cv,cv)
        forecast=self.cache[day]
        if forecast['signal_date']>=day:raise AssertionError('Future information')
        mu,cov=forecast[self.mode];current=np.array([actual_weights.get(s,0.) for s in ASSETS])
        w=optimise(mu,cov,current,self.penalty)
        w=cap_portfolio_vol(dict(zip(ASSETS,w)),snapshot.covariance,.1)
        target={s:w[s]*.995 for s in ASSETS};tv=estimated_vol(target,snapshot.covariance)
        drift=max(abs(target[s]-actual_weights.get(s,0)) for s in ASSETS)
        trade=drift>=.005 and (weekly or tv<cv)
        return Decision(snapshot.date,day.date(),-1,target,trade,'weekly-optimisation' if weekly else 'risk-cut',tv,cv)
