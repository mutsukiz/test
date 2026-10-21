import warnings;warnings.filterwarnings("ignore")
import os,sys
from refine import *
import matplotlib;matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.ndimage import map_coordinates,gaussian_filter
import report
from tracer import paraxial
from aimed import _core
from lenssim_export import TEMPLATE,bfd_obj
from lenssim_io import write
OUT="/home/user/test/acrylic_ball_lens"; os.makedirs(OUT,exist_ok=True)
NDV={"PMMA":(1.4917,57.4),"N-SF5":(1.67270,32.25),"N-SF11":(1.78472,25.68)}
HIMG=14.2; SENSOR=(23.6,15.6)
def design(fno):
    lay,v=refine(80,["48-233","47-917"],[0,0],fno)
    return lay,v
def lenssim(lay,v,fno,path):
    c,t,n,kind,stop=lay.build(v)
    efl,bfd,zep=paraxial(c,t,n,stop,1)
    _,ys=_core(c,t,n,np.zeros(1),np.array([efl/fno/2]),np.full(1,zep),np.zeros(1),np.zeros(1),np.ones(1),np.array([1]),upto=stop)
    a_stop=float(abs(ys[0]))
    glasses=lay.g;dias=[80,30,25];elems=[];gi=0;k=1  # k=0 is the stop
    while k<len(c):
        f,r=k,k+1
        gap=t[r] if r<len(c)-1 else 0.0
        nd,V=NDV[glasses[gi]]
        elems.append(dict(R1=100000.0 if abs(c[f])<1e-12 else 1/c[f],R2=100000.0 if abs(c[r])<1e-12 else 1/c[r],ct=t[f],gap_after=gap,nd=nd,V=V,dia1=dias[gi],dia2=dias[gi]))
        gi+=1;k=r+1
    title="Acrylic ball 80mm + Edmund 48-233 + 47-917, f%.0fmm F%.1f (APS-C)"%(efl,fno)
    header=[("link",title),*TEMPLATE,("f s pos start","%.7f"%bfd_obj(c,t,n,1000.0)),("f s pos start m","1.0"),("f s pos end","%.7f"%bfd_obj(c,t,n,500000.0)),("f s pos end m","500.0"),
            ("image scale ref","%.7f"%(5.885/efl)),("default sensor size","0.0426"),("squeeze factor","1.0"),("chromatic aberration","1.0"),("f number","%.1f"%fno),("unit scale","0.001"),
            ("aperture idx","0"),("aperture r","%.7f"%a_stop),("aperture d","%.7f"%t[0])]
    write(path,header,elems,0,t[0],None)
    # re-read the written file and recompute paraxial EFL/BFD
    kv=dict(l.split(" = ",1) for l in open(path,encoding="utf-8").read().splitlines() if " = " in l)
    E=len(elems);cc=[];tt=[];nn=[]
    for e in range(E):
        cc+=[1/float(kv["r%d"%(2*e+1)]),1/float(kv["r%d"%(2*e+2)])];tt+=[float(kv["d%d"%(2*e+1)]),float(kv["d%d"%(2*e+2)]) if e<E-1 else bfd]
    nds=[float(kv["ior%d"%(e+1)]) for e in range(E)];y,u,n0=1.0,0.0,1.0
    for k2 in range(2*E):
        n1=nds[k2//2] if k2%2==0 else 1.0;u=(n0*u-y*cc[k2]*(n1-n0))/n1;n0=n1
        if k2<2*E-1:y+=u*tt[k2]
    print("Lens Sim file check: EFL from file %.3f (design %.3f), a_stop %.3f"%(-1/u,efl,a_stop))
def layout(lay,v,fno,path):
    c,t,n,kind,stop=lay.build(v);efl,bfd,zep=paraxial(c,t,n,stop,1)
    z=np.concatenate([[0],np.cumsum(t)]);a=float(abs(_core(c,t,n,np.zeros(1),np.array([efl/fno/2]),np.full(1,zep),np.zeros(1),np.zeros(1),np.ones(1),np.array([1]),upto=stop)[1][0]))
    fig,ax=plt.subplots(figsize=(13,5.2))
    dias=[None,80,80,30,30,25,25]
    def prof(k):
        R=1/c[k] if c[k] else np.inf;h=np.linspace(-dias[k]/2*0.999,dias[k]/2*0.999,200)
        s=np.zeros_like(h) if not np.isfinite(R) else R-np.sign(R)*np.sqrt(R*R-h*h);return z[k]+s,h
    cols=["#cde7ff","#ffe2c0","#d6ffd6"]
    for e,(f,r) in enumerate(((1,2),(3,4),(5,6))):
        zf,hf=prof(f);zr,hr=prof(r);ax.fill(np.concatenate([zf,zr[::-1]]),np.concatenate([hf,hr[::-1]]),fc=cols[e],ec="k",lw=1)
    ax.plot([z[0]]*2,[a,a+8],"k",lw=3);ax.plot([z[0]]*2,[-a,-a-8],"k",lw=3)
    zi=z[-1];ax.plot([zi,zi],[-11.8,11.8],"k",lw=2)
    SPEC["fno"]=fno
    fcol={0:"#2b6cff",1:"#22a35a",2:"#e0402c"}
    for ang,col in ((0,"tab:blue"),(6,"tab:green"),(12,"tab:red")):
        th=np.radians(ang)
        for fr in (-1,0,1):
            p=np.array([[0,fr*a*0.98,0.0]]);d=np.array([[0,np.sin(th),np.cos(th)]]);pts=[(0.0,p[0,1])];n0=1.0
            ok=True;zz=z[0]
            for k in range(1,len(c)):
                Rc=np.inf if c[k]==0 else 1/c[k];n1=n[k][1]
                p,nrm=sph(p,d,z[k],Rc);nrm=nrm*np.where((nrm*d).sum(1)>0,-1,1)[:,None];d=refr(d,nrm,n0,n1);n0=n1;pts.append((p[0,2],p[0,1]))
            tt_=(zi-p[0,2])/d[0,2];pts.append((zi,p[0,1]+tt_*d[0,1]))
            pz,py=zip(*pts);ax.plot(pz,py,color=col,lw=0.8)
        # entering segment from the left
    ax.set_aspect("equal");ax.set_xlim(-8,zi+4);ax.set_ylim(-45,45);ax.set_xlabel("z [mm]  (object side → sensor)");ax.set_ylabel("height [mm]")
    ax.set_title("Ball Φ80 + #48-233 + #47-917 | f=%.0f mm F/%.1f | rays at 0°, 6°, 12° (d-line) | stop Φ%.1f at left | sensor at right"%(efl,fno,2*a))
    ax.grid(alpha=.25);fig.tight_layout();fig.savefig(path,dpi=130);plt.close(fig)
def simulate(lay,v,fno,path,title):
    SPEC.update(fno=fno)
    fr,yc,rms=report.channel_maps(lay,v)
    W,H=2360,1560;px=SENSOR[0]/W
    yy,xx=np.mgrid[0:H,0:W];X=(xx-W/2+.5)*px;Y=(yy-H/2+.5)*px
    g=2.0;chart=np.clip(((np.abs((X+50)%g-g/2)<0.05)|(np.abs((Y+50)%g-g/2)<0.05))*0.9,0,1)
    rng=np.random.default_rng(3)
    stars=np.zeros((H,W))
    for _ in range(260):
        i,j=rng.integers(40,H-40),rng.integers(40,W-40);stars[i-2:i+3,j-2:j+3]=rng.uniform(.5,1)
    rimg=np.sqrt(X**2+Y**2);outs=[]
    for scene,nm in ((chart,"grid chart"),(stars,"point lights")):
        out=np.zeros((H,W,3))
        for ch,w in ((2,0),(1,1),(0,2)):
            r_obj=np.interp(rimg,yc[:,w],yc[:,1],right=np.nan);s=np.where(rimg>0,r_obj/np.maximum(rimg,1e-9),1.0)
            col=X*s/px+W/2-.5;row=Y*s/px+H/2-.5
            img=np.nan_to_num(map_coordinates(scene,[row,col],order=1,cval=0.0))
            sig=np.interp(rimg,yc[:,w],rms[:,w])/px/np.sqrt(2)
            levels=[0.3,0.7,1.5,2.5,4,6];stack=[gaussian_filter(img,l) for l in levels]
            idx=np.clip(np.interp(sig,levels,np.arange(len(levels))),0,len(levels)-1);lo=np.floor(idx).astype(int);hi=np.minimum(lo+1,len(levels)-1);f_=idx-lo
            res=np.zeros_like(img)
            for k in range(len(levels)): res+=np.where(lo==k,1-f_,0)*stack[k]+np.where(hi==k,f_,0)*stack[k]
            out[...,ch]=res
        outs.append(np.clip(out*(1.2 if nm=="grid chart" else 3.0),0,1))
    fig=plt.figure(figsize=(13,10.5));gs=fig.add_gridspec(3,3,height_ratios=[2.3,2.3,1.2])
    for r_,(o,nm) in enumerate(zip(outs,("grid chart","point lights"))):
        ax=fig.add_subplot(gs[r_,:]);ax.imshow(o,extent=[-SENSOR[0]/2,SENSOR[0]/2,SENSOR[1]/2,-SENSOR[1]/2]);ax.set_title("%s — %s"%(title,nm),fontsize=10);ax.set_xlabel("mm on sensor (APS-C 23.6×15.6)")
    for j,(cx,cy,nm) in enumerate(((1,1,"centre"),(5,3,"mid"),(11,7,"corner"))):
        a=fig.add_subplot(gs[2,j]);c0=int((cx)/px+W/2);r0=int(cy/px+H/2);hw=90
        a.imshow(outs[0][r0-hw:r0+hw,c0-hw:c0+hw],interpolation="nearest");a.set_title("grid crop: %s (%.1f,%.1f) mm"%(nm,cx,cy),fontsize=9);a.axis("off")
    fig.tight_layout();fig.savefig(path,dpi=100);plt.close(fig)
if __name__=="__main__":
    for fno in (5.6,8):
        lay,v=design(fno);tag="F%s"%str(fno).replace(".","_")
        lenssim(lay,v,fno,OUT+"/BallLens_%s_lenssim.txt"%tag)
        layout(lay,v,fno,OUT+"/BallLens_%s_layout.png"%tag)
        SPEC.update(fno=fno)
        report.spot_figure(lay,v,OUT+"/BallLens_%s_spots.png"%tag,"Ball80 + 48-233 + 47-917, F%g: spots (F/d/C)"%fno,box=0.12)
        simulate(lay,v,fno,OUT+"/BallLens_%s_simulation.png"%tag,"Ball80 lens F%g, simulated"%fno)
        rows,efl,bfd=report.prescription(lay,v)
        with open(OUT+"/BallLens_%s_prescription.csv"%tag,"w") as f:
            f.write("surface,radius_mm,thickness_mm,glass,nd\n")
            for r in rows:f.write("%d,%s,%.3f,%s,%.5f\n"%(r["surf"],"inf" if not np.isfinite(r["R"]) else "%.3f"%r["R"],r["t"],r["glass"],r["nd"]))
