import sys,os,json,time
ROOT='/tmp/claude-0/-home-user-test/899f3e06-8ae0-5310-8886-a5112a7c820a/scratchpad/lh2/lens_handover'
sys.path.insert(0,ROOT+'/lenslab')
import numpy as np
import tracer, opt, report, metrics
from tracer import IDX, paraxial
IDX["PMMA"]=np.array([1.49776,1.49170,1.48920])
IDX["N-SF11"]=IDX["N-SF11"]
from catalog import PARTS
from catopt import CatLayout, optimize_gaps
from aimed import make_tracer
from opt import SPEC
BALL_D=60.0
def add_ball(D):
    R=D/2
    PARTS.append(dict(typ="BALL",stock="BALL%d"%D,dia=D,f=0,glass="PMMA",c1=1/R,c2=-1/R,ct=D,sa=D/2*0.95))
    return len(PARTS)-1
BALL=add_ball(60)
def set_trace(stop):
    base=make_tracer(lambda c,t,n: stop)
    fn=lambda c,t,n,epd,zep,f,rings=6,img_extra=0.0: base(c,t,n,epd,zep,f,rings=max(rings,6))
    opt.trace_real=fn; report.trace_real=fn; metrics.trace_real=fn
def evaluate(parts,flips,stop_after,fno,efl=50.0,bfd_min=8.0,x0=None,nfev=150):
    SPEC.update(efl=efl,fno=fno,hfov=float(np.degrees(np.arctan(21.6/efl))),maxsa=30.0,bfd_min=bfd_min,tt_max=200.0)
    lay=CatLayout(parts,flips,stop_after,0.0)
    v0=np.array([1.0]*lay.E+[30.0]) if x0 is None else x0
    c,t,n,kind,stop=lay.build(v0); set_trace(stop)
    v,m=optimize_gaps(lay,v0,nfev=nfev)
    return lay,v,m
if __name__=="__main__":
    pid={p["stock"]:i for i,p in enumerate(PARTS)}
    t0=time.time()
    lay,v,m=evaluate([BALL,pid["47-913"]],[0,0],1,8.0)
    print(time.time()-t0,m,v)
    d=lay.residuals(v,detail=True);print({k:d[k] for k in("efl","axial_FC_mm","tca_mm","spot_um","dist_pct")})
