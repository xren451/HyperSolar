import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Ellipse, FancyBboxPatch, FancyArrowPatch, Polygon, Rectangle
from matplotlib.lines import Line2D
import numpy as np

mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Noto Sans CJK SC", "Arial", "DejaVu Sans"],
    "svg.fonttype": "none",
    "pdf.fonttype": 42,
    "font.size": 9,
    "axes.linewidth": 0.7,
})

COL = {
    "ink": "#243746", "muted": "#667987", "line": "#AAB7C2",
    "sat": "#279A8A", "nwp": "#8172B2", "field": "#4F83CC",
    "site": "#D9822B", "cross": "#9271B8", "local": "#E28B62",
    "global": "#54768C", "pale": "#F6F8FA", "white": "#FFFFFF",
}

fig = plt.figure(figsize=(14.0, 8.3), facecolor="white")
fig.text(0.045, 0.965, "Hypergraph nodes and edge types for cross-scale cloud-state exchange",
         ha="left", va="top", fontsize=18, fontweight="bold", color=COL["ink"])
fig.text(0.045, 0.925,
         "Toy region: gridded cloud–radiative states interact with a site history and future queries",
         ha="left", va="top", fontsize=10.5, color=COL["muted"])

gs = fig.add_gridspec(2, 2, left=0.045, right=0.97, top=0.88, bottom=0.13,
                      hspace=0.22, wspace=0.12)
axes = [fig.add_subplot(gs[0,0]), fig.add_subplot(gs[0,1]),
        fig.add_subplot(gs[1,0]), fig.add_subplot(gs[1,1])]
for ax in axes:
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")

def panel(ax, letter, title, subtitle):
    ax.add_patch(FancyBboxPatch((0,0),1,1,boxstyle="round,pad=0.012,rounding_size=0.025",
                 facecolor=COL["white"],edgecolor="#D9E1E7",linewidth=1.0,zorder=-10))
    ax.text(0.035,0.95,letter,fontsize=13,fontweight="bold",color=COL["field"],va="top")
    ax.text(0.105,0.95,title,fontsize=11.4,fontweight="bold",color=COL["ink"],va="top")
    ax.text(0.105,0.885,subtitle,fontsize=8.2,color=COL["muted"],va="top")

def node(ax, x, y, color, label=None, r=0.035, marker="o", fc=None, lw=1.2, z=5):
    if marker == "*":
        ax.scatter([x],[y],s=180,c=[color],marker="*",edgecolors="white",linewidths=0.8,zorder=z)
    else:
        ax.add_patch(Circle((x,y),r,facecolor=fc or color,edgecolor=color,linewidth=lw,zorder=z))
    if label:
        ax.text(x,y,label,ha="center",va="center",fontsize=6.4,color="white" if marker=="o" else COL["ink"],zorder=z+1)

def arrow(ax, p1, p2, color=COL["muted"], lw=1.0, style="-|>", rad=0):
    ax.add_patch(FancyArrowPatch(p1,p2,arrowstyle=style,mutation_scale=9,linewidth=lw,
                 color=color,connectionstyle=f"arc3,rad={rad}",zorder=3))

# A — Node vocabulary and space/time placement
ax=axes[0]; panel(ax,"A","Node types","Nodes are states tied to a source, location and time.")
gx0, gy0, dx, dy = 0.15, 0.36, 0.115, 0.105
for r in range(3):
    for c in range(3):
        x,y=gx0+c*dx,gy0+(2-r)*dy
        ax.add_patch(Rectangle((x-0.047,y-0.038),0.094,0.076,facecolor="#F6F9FC",
                     edgecolor="#DCE4EB",linewidth=0.7,zorder=0))
        node(ax,x,y,COL["field"],"q",r=0.026)
ax.text(0.38,0.56,"future field query  qᶠᵢ,τ",fontsize=8,color=COL["field"],va="center")
node(ax,0.20,0.28,COL["site"],"s",marker="*",z=6)
ax.text(0.38,0.28,"site-history / ramp node  hˢ",fontsize=8,color=COL["site"],va="center")
node(ax,0.20,0.19,COL["sat"],"S",r=0.026)
ax.text(0.38,0.19,"satellite-history node  hˢᵃᵗᵢ",fontsize=8,color=COL["sat"],va="center")
node(ax,0.20,0.10,COL["nwp"],"N",r=0.026)
ax.text(0.38,0.10,"NWP-history node  hⁿʷᵖᵢ",fontsize=8,color=COL["nwp"],va="center")
ax.text(0.70,0.53,"history",fontsize=7.5,color=COL["muted"],ha="center")
arrow(ax,(0.73,0.50),(0.89,0.50),COL["muted"],0.9)
ax.text(0.91,0.53,"future",fontsize=7.5,color=COL["muted"],ha="center")
ax.text(0.70,0.42,"Only future queries are updated; history is read as context.",
        fontsize=7.1,color=COL["muted"],ha="left",wrap=True)

# B — same-cell cross-source hyperedge
ax=axes[1]; panel(ax,"B","Same-cell cross-source hyperedge","One relation joins multiple sources at the same grid cell.")
coords=[(0.27,0.59),(0.50,0.59),(0.73,0.59)]
colors=[COL["sat"],COL["nwp"],COL["field"]]
labels=["satellite\nhistory","NWP\nhistory","future field\nquery"]
ax.add_patch(Ellipse((0.50,0.59),0.68,0.36,facecolor=COL["cross"],alpha=0.10,
                     edgecolor=COL["cross"],linewidth=2.0,zorder=1))
for (x,y),c,l in zip(coords,colors,labels):
    node(ax,x,y,c,r=0.060)
    ax.text(x,y-0.10,l,ha="center",va="top",fontsize=7.6,color=COL["ink"])
ax.text(0.50,0.82,"Eᶜʳᵒˢˢᵢ = {hˢᵃᵗᵢ, hⁿʷᵖᵢ, qᶠᵢ,τ}",ha="center",fontsize=9.1,
        color=COL["cross"],fontweight="bold")
ax.text(0.50,0.17,"Interpretation: observed cloud context + atmospheric forcing jointly inform\nthe future state of this grid cell.",
        ha="center",va="center",fontsize=7.8,color=COL["muted"])

# C — ramp-conditioned local site-neighborhood hyperedge
ax=axes[2]; panel(ax,"C","Ramp-conditioned site–neighborhood hyperedge","A site event modulates messages to nearby future grid states.")
cx0,cy0,dx,dy=0.24,0.34,0.115,0.105
for r in range(3):
    for c in range(3):
        x,y=cx0+c*dx,cy0+(2-r)*dy
        ax.add_patch(Rectangle((x-0.045,y-0.037),0.09,0.074,facecolor="#F8FAFC",
                     edgecolor="#DCE4EB",linewidth=0.65,zorder=0))
        node(ax,x,y,COL["field"],"q",r=0.022)
# Local hyperedge includes a 3x3 illustrative neighborhood and site nodes.
ax.add_patch(Ellipse((cx0+dx,cy0+dy),0.40,0.34,facecolor=COL["local"],alpha=0.12,
                     edgecolor=COL["local"],linewidth=2.0,linestyle="--",zorder=1))
node(ax,cx0+dx,cy0+dy,COL["site"],"s",marker="*",z=7)
ax.text(0.64,0.68,"ramp attributes on this hyperedge",fontsize=8.0,fontweight="bold",color=COL["site"])
ax.add_patch(FancyBboxPatch((0.64,0.43),0.30,0.19,boxstyle="round,pad=0.012,rounding_size=0.02",
             facecolor="#FFF8F0",edgecolor="#E9C39E",linewidth=0.8))
ax.text(0.79,0.525,"amplitude  •  duration\nslope  •  direction  •  quality",ha="center",va="center",
        fontsize=7.4,color=COL["ink"])
arrow(ax,(0.64,0.48),(0.54,0.45),COL["site"],1.2)
ax.text(0.64,0.31,r"$A_{v,e_s}=\mathrm{softmax}_{v\in\mathcal{N}(s)} f(h_v,a_s^{ramp},d_{vs})$",
        fontsize=7.2,color=COL["ink"])
ax.text(0.64,0.21,"Ramp is computed from observed history only.\nIt weights site-to-neighborhood exchange; it is not a future label.",
        fontsize=7.0,color=COL["muted"],va="top")

# D — global edge and message passing
ax=axes[3]; panel(ax,"D","Global regional hyperedge","The global edge carries broad cloud–radiative context across the field.")
points=[]
for r in range(3):
    for c in range(3):
        x,y=0.24+c*0.12,0.39+(2-r)*0.11
        points.append((x,y))
ax.add_patch(Ellipse((0.48,0.50),0.50,0.50,facecolor=COL["global"],alpha=0.10,
                     edgecolor=COL["global"],linewidth=2.0,zorder=1))
for i,(x,y) in enumerate(points): node(ax,x,y,COL["field"],"q",r=0.022)
node(ax,0.48,0.50,COL["global"],"e",r=0.040,z=8)
node(ax,0.54,0.26,COL["site"],"s",marker="*",z=8)
arrow(ax,(0.62,0.50),(0.78,0.50),COL["global"],1.0)
ax.text(0.84,0.54,"aggregate",fontsize=7.6,color=COL["global"],ha="center")
ax.text(0.84,0.43,"update",fontsize=7.6,color=COL["global"],ha="center")
arrow(ax,(0.78,0.46),(0.62,0.46),COL["global"],1.0)
ax.text(0.50,0.14,"Node → hyperedge → node: one hyperedge can coordinate\nmessages among many grid cells and sites in a single operation.",
        ha="center",va="center",fontsize=7.7,color=COL["muted"])

# Shared legend strip
handles=[Line2D([0],[0],marker='o',color='none',markerfacecolor=COL[k],markeredgecolor=COL[k],markersize=7,label=lab)
         for k,lab in [("sat","Satellite history"),("nwp","NWP history"),("field","Future grid query")]]
handles.append(Line2D([0],[0],marker='*',color='none',markerfacecolor=COL["site"],markeredgecolor=COL["site"],markersize=10,label="Site history / ramp"))
fig.legend(handles=handles,loc="lower left",bbox_to_anchor=(0.05,0.052),ncol=4,frameon=False,
           fontsize=8.1,handletextpad=0.45,columnspacing=1.8)
fig.text(0.97,0.055,"Conceptual schematic; hyperedge memberships shown for illustration.",
         ha="right",va="center",fontsize=7.2,color=COL["muted"])

out="hypergraph_nodes_and_edges"
fig.savefig(out+".svg",bbox_inches="tight")
fig.savefig(out+".pdf",bbox_inches="tight")
fig.savefig(out+".png",dpi=300,bbox_inches="tight")
