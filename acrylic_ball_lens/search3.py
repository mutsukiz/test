import warnings;warnings.filterwarnings("ignore")
import sys,json
from multiprocessing import Pool
from ballsys import *
D=float(sys.argv[1]);FNO=float(sys.argv[2]);OUT=sys.argv[3];HIMG=14.2
pid0={p["stock"]:i for i,p in enumerate(PARTS)}
PARTS[pid0["62-574"]].update(c1=1/45.94,c2=0.0,ct=4.0)
BALLD=add_ball(int(D))
fb=1.4917*D/(4*0.4917);EFL=1.1*fb
cand=[i for i,p in enumerate(PARTS) if p["typ"]!="BALL" and p["dia"]>=15]
neg=[i for i in cand if PARTS[i]["f"]<0]
neg_fl=[i for i in neg if PARTS[i]["glass"] in("N-SF11","N-SF5")]
pos=[i for i in cand if 25<=PARTS[i]["f"]<=150]
def variants(i): return [(i,0)] if PARTS[i]["typ"] in("DCX","DCV") else [(i,0),(i,1)]
jobs=[]
for x in neg_fl:
  for xv,xf in variants(x):
    for y in pos+neg:
      for yv,yf in variants(y):
        jobs.append(([BALLD,xv,yv],[0,xf,yf],0)); jobs.append(([BALLD,yv,xv],[0,yf,xf],0))
def run(job):
    parts,flips,sa=job
    SPEC.update(efl=EFL,fno=FNO,hfov=float(np.degrees(np.arctan(HIMG/EFL))),maxsa=60.0,bfd_min=8.0,tt_max=400.0)
    try:
        lay=CatLayout(parts,flips,sa,0.0)
        v0=np.array([3.0]*lay.E+[EFL*0.6]);c,t,n,kind,stop=lay.build(v0);set_trace(stop)
        v,m=optimize_gaps(lay,v0,nfev=60)
        d=lay.residuals(v,detail=True)
        return dict(stocks=[PARTS[i]["stock"] for i in parts],flips=flips,sa=sa,x=v.tolist(),merit=m,efl=float(d["efl"]),ax=float(d["axial_FC_mm"]),tca=[float(a) for a in d["tca_mm"]],spot=d["spot_um"],tt=float(d["tt"]))
    except Exception: return None
if __name__=="__main__":
    with Pool(4) as p: res=[r for r in p.imap_unordered(run,jobs,chunksize=8) if r and np.isfinite(r["merit"])]
    res.sort(key=lambda r:r["merit"]); json.dump(res,open(OUT,"w"))
    for r in res[:12]: print(r["stocks"],r["flips"],"merit %.3g efl %.1f ax %.3f"%(r["merit"],r["efl"],r["ax"]),"tca",np.round(np.array(r["tca"])*1000).astype(int),"spot",[[round(a) for a in s] for s in r["spot"]],np.round(r["x"],2))
