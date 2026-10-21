import warnings;warnings.filterwarnings("ignore")
import sys
from ballsys import *
from flat import sph, refr
pid={p["stock"]:i for i,p in enumerate(PARTS)}
def indep(c,t,n,a,ang,iw,nr=16):
    g=np.linspace(-1,1,2*nr+1);X,Y=np.meshgrid(g,g);m=X**2+Y**2<=1
    x=X[m]*a;y=Y[m]*a;th=np.radians(ang)
    p=np.stack([x,y,np.zeros_like(x)],1);d=np.tile([0,np.sin(th),np.cos(th)],(len(x),1))
    z=t[0];n0=1.0
    for k in range(1,len(c)):
        Rc=np.inf if c[k]==0 else 1/c[k];n1=n[k][iw]
        p,nrm=sph(p,d,z,Rc);nrm=nrm*np.where((nrm*d).sum(1)>0,-1,1)[:,None];d=refr(d,nrm,n0,n1);n0=n1;z+=t[k]
    tt=(z-p[:,2])/d[:,2];q=p+tt[:,None]*d;return q[:,:2]
def refine(D,stocks,flips,fno,himg=14.2,gmin=0.5,efl_k=1.1):
    b=add_ball(D);fb=1.4917*D/(4*0.4917);efl=efl_k*fb
    SPEC.update(efl=efl,fno=fno,hfov=float(np.degrees(np.arctan(himg/efl))),maxsa=60.0,bfd_min=8.0,tt_max=400.0)
    parts=[b]+[pid[s] for s in stocks];lay=CatLayout(parts,[0]+flips,0,0.0)
    lo,hi=lay.bounds();lo=lo.copy();lo[0]=0.3;lo[1]=gmin;lo[2]=0.5;lay.bounds=lambda:(lo,hi)
    best=None
    for g in (0.6,2,5):
        v0=np.array([0.4,gmin+0.1,g,efl*0.17]);c,t,n,kind,stop=lay.build(v0);set_trace(stop)
        v,m=optimize_gaps(lay,v0,nfev=250)
        if v is not None and (best is None or m<best[1]): best=(v,m)
    v,m=best;c,t,n,kind,stop=lay.build(v);set_trace(stop);d=lay.residuals(v,detail=True)
    print(D,stocks,flips,"F/%g EFL %.1f himg %.1f hfov %.1f ax %.3f tca %s dist %s"%(fno,d["efl"],himg,SPEC["hfov"],d["axial_FC_mm"],np.round(np.array(d["tca_mm"])*1000).astype(int),np.round(d["dist_pct"],1)))
    print("  RMS(F,d,C):",[[round(a) for a in s] for s in d["spot_um"]],"hmax",np.round(d["hmax"],1),"gaps",np.round(v,2))
    for q in (p for p in parts[1:]): 
        P=PARTS[q];print("  ",P["stock"],P["typ"],P["glass"],"dia",P["dia"],"R1 %.2f R2 %s ct %.2f"%(1/P["c1"],"inf" if P["c2"]==0 else "%.2f"%(1/P["c2"]),P["ct"]))
    a=d["hmax"][0]
    xs=[]
    for ang in np.degrees(np.arctan(himg/efl))*np.array([0,.5,.75,1]):
        r=[];
        for iw in (0,1,2):
            q=indep(c,t,n,a,ang,iw);r.append(np.sqrt(((q-q.mean(0))**2).sum(1).mean())*1000)
        xs.append([round(z) for z in r])
    print("  indep RMS:",xs)
    return lay,v
if __name__=="__main__":
    for st,fl in ((["48-233","47-917"],[0,0]),(["48-234","47-917"],[0,0])):
        for f in (5.6,8): refine(80,st,fl,f)
