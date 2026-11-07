"""Deterministic SVG figures from measured CSVs; no environment upgrades."""
from pathlib import Path
from html import escape
import json
import numpy as np
import pandas as pd
from .checkpoint2_protocol import sha


def line_svg(path,series,title,x_label,source):
    points=[(float(x),float(y)) for _,xs,ys in series for x,y in zip(xs,ys)]
    if not points:return
    if not np.isfinite(points).all():raise ValueError('Cannot plot nonfinite measurements')
    xmax=max(x for x,_ in points) or 1.;ymax=max(y for _,y in points)*1.08 or 1.
    width,height=680,320;left,right,top,bottom=70,25,65,55
    def px(x):return left+x/xmax*(width-left-right)
    def py(y):return height-bottom-y/ymax*(height-top-bottom)
    xml=['<svg xmlns="http://www.w3.org/2000/svg" width="680" height="320" viewBox="0 0 680 320">',
         '<rect width="680" height="320" fill="white"/>',
         '<g font-family="Arial" font-size="12" fill="black">',
         f'<text x="70" y="20" font-size="15">{escape(title)}</text>',
         f'<desc>Source {escape(Path(source).name)} SHA256 {sha(source)}; seed means where grouped</desc>']
    for x in np.linspace(0,xmax,6):
        xml.append(f'<path d="M {px(x):.3f} {top} V {height-bottom}" stroke="#dddddd"/><text x="{px(x):.3f}" y="{height-bottom+19}" text-anchor="middle">{x:.2g}</text>')
    for y in np.linspace(0,ymax,5):
        xml.append(f'<path d="M {left} {py(y):.3f} H {width-right}" stroke="#dddddd"/><text x="{left-8}" y="{py(y)+4:.3f}" text-anchor="end">{y:.3f}</text>')
    for i,(name,xs,ys) in enumerate(series):
        color=['#235789','#bd3f36','#38734a','#6c4e91'][i%4]
        coords=' '.join(f'{px(float(x)):.3f},{py(float(y)):.3f}' for x,y in zip(xs,ys))
        xml.append(f'<polyline points="{coords}" fill="none" stroke="{color}" stroke-width="2"/>')
        xml.append(f'<text x="{left+i*175}" y="45" fill="{color}">{escape(str(name))}</text>')
    xml.extend([f'<text x="{width/2}" y="{height-8}" text-anchor="middle">{escape(x_label)}</text>',
                '<text x="12" y="160" transform="rotate(-90 12 160)" text-anchor="middle">Validation / Test NDCG@10</text>',
                '</g></svg>'])
    path.parent.mkdir(parents=True,exist_ok=True);path.write_text('\n'.join(xml),encoding='utf-8')


def write_figures(root):
    root=Path(root);source=root/'K3_validation_epochs.csv'
    if source.exists():
        f=pd.read_csv(source)
        line_svg(root/'figures/K3_validation.svg',[(n,g.epoch,g.val_ndcg_10) for n,g in f.groupby('job',sort=True)],
                 root.name+' K3 validation by epoch','Epoch (0 is pre-training)',source)
    source=root/'robustness.csv'
    if source.exists():
        f=pd.read_csv(source)
        for kind,g in f.groupby('mechanism'):
            curves=[]
            for name,rows in g.groupby('source'):
                means=rows.groupby('missing_rate').ndcg_10.mean().sort_index()
                curves.append((name,means.index,means.to_numpy()))
            line_svg(root/f'figures/missingness_{kind}.svg',curves,root.name+' '+kind+' missingness (seed mean)',
                     'Fraction of available source items masked',source)
