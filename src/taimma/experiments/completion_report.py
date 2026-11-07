"""Seven-table report from measured registry/derived CSVs, never typed numbers."""
from pathlib import Path
import json,zipfile
import numpy as np
import pandas as pd
from .checkpoint2_protocol import write_json,read_ranks
from .mini_checkpoint_report import _table,_paragraph,NS


def read_json(p,default=None):return json.loads(p.read_text(encoding='utf-8')) if p.exists() else default


def write_report(root,config):
    root=Path(root)
    if not (root/'results.csv').exists():raise FileNotFoundError('Export the measured registry first')
    registry=pd.read_csv(root/'results.csv');tables=[]
    def values(names,part='test'):
        rows=registry[registry.job.isin(names)]
        return rows[part+'_ndcg_10'].to_numpy() if len(rows)==len(names) else []
    def summary(v):
        return f'{np.mean(v):.6f} ± {np.std(v,ddof=1):.6f}' if len(v)>=3 else 'Chưa đủ 3 seed'
    def jobs(kind,lab):return [f'id_{s}' if lab=='ID' else f'{kind}_{lab}_{s}' for s in [42,43,44]]
    rows=[]
    for lab in ['ID','C','B','T','CB','CT','BT','CBT']:
        rows.append([lab,summary(values(jobs('early',lab))),summary(values(jobs('late',lab)))])
    for src,without in [('C','BT'),('B','CT'),('T','CB')]:
        cells=[]
        for kind in ['early','late']:
            a,b,c,d=[values(jobs(kind,k)) for k in [src,'ID','CBT',without]]
            cells.append(f'{np.mean(a-b):+.6f} / {np.mean(c-d):+.6f}' if all(len(v)==3 for v in [a,b,c,d]) else 'Chưa có')
        rows.append(['Add / Drop '+src]+cells)
    tables.append(('1 Chẩn đoán B',['Nguồn','Early Test mean ± SD','Late Test mean ± SD'],rows,[1,3,3],
                   'Add = một nguồn − ID; Drop = CBT − bỏ nguồn. SD mẫu qua seed; không gán ý nghĩa seed-level cho CI người dùng.'))
    controls=[]
    from .study_completion import paired_summary
    def ranks(name,part='test'):
        rec=read_json(root/'jobs'/name/'completed.json')
        return read_ranks((root/'jobs'/name/rec['checkpoint']).parent/f'per_user_{part}.csv')
    for source in ['category','brand','title']:
        for kind in ['shuffled','random']:
            names=[f'control_{kind}_{source}_{s}' for s in [42,43,44]]
            if len(values(names))==3 and len(values(jobs('early','CBT')))==3:
                ci=paired_summary({s:ranks(n) for s,n in zip([42,43,44],names)},
                                  {s:ranks(f'early_CBT_{s}') for s in [42,43,44]},config['bootstrap_reps'])
                text=f"{ci['delta_left_minus_right']:+.6f} [{ci['ci95'][0]:+.6f}; {ci['ci95'][1]:+.6f}]"
            else:text='Chưa có'
            controls.append([source+' '+kind,summary(values(names)),text])
    tables.append(('2 Đối chứng khi huấn luyện',['Nguồn thay thế','Test mean ± SD','Δ đối chứng − thật [CI95%]'],controls,[1.7,2,3.3],
                   'Random V2: C/B đều trên từ vựng; Title Gaussian cùng mean/covariance thật. CI percentile ghép cặp người dùng, 1.000 lần; chưa hiệu chỉnh nhiều so sánh.'))
    diagnostic=root/'historical_diagnostics';k1=read_json(diagnostic/'K1.json');rows=[]
    if k1:
        rows.append(['K1',f"CE ID → late: base {k1['train_base']['mean_ce_id']:.6f} → {k1['train_base']['mean_ce_late']:.6f}; meta {k1['train_meta']['mean_ce_id']:.6f} → {k1['train_meta']['mean_ce_late']:.6f}"])
    else:rows.append(['K1','Chưa có diagnostics đã nhập'])
    for job in ['K2','K3','K4']:
        f=registry[registry.job==job]
        if len(f):
            r=f.iloc[0];text=f"Val/Test {r.val_ndcg_10:.6f}/{r.test_ndcg_10:.6f}; best epoch {int(r.best_epoch)}{' †' if r.best_epoch_le5 else ''}; epoch 0→1: {r.initial_val:.6f}→{r.epoch1_val:.6f}"
        else:text='Chưa có'
        if job=='K4':
            joint=registry[registry.job.isin(['K4','late_joint_43','late_joint_44'])]
            if len(joint)==3:text+='; K4 3 seed Val '+summary(joint.val_ndcg_10.to_numpy())+'; Test '+summary(joint.test_ndcg_10.to_numpy())
        rows.append([job,text])
    tables.append(('3 Nguyên nhân hợp nhất muộn',['Mục','Số liệu seed 42 CBT'],rows,[.5,6.5],
                   '†Best epoch ≤5; đối chiếu ID và late_reference_42 trong results.csv. K3_validation_epochs.csv chứa toàn bộ đường Val; K4 nhiều seed được ghi riêng, không suy SD từ một seed.'))
    group=read_json(root/'group_analysis.json',[]);rows=[]
    for kind in ['early','late']:
        g=[r for r in group if r['mechanism']==kind]
        rows.append([kind,summary([r['global_test'] for r in g]),summary([r['group_val_selected_test'] for r in g]),summary([r['group_test_oracle'] for r in g])])
    tables.append(('4 Lựa chọn theo nhóm',['Cơ chế','Global Test','Nhóm Val → Test','Oracle Test'],rows,[1,2,2,2],
                   'Cùng nhóm lịch sử được khóa trên Val cho cả hai cơ chế. Chọn subset bằng Val; oracle Test chỉ là trần mô tả, không dùng chọn backbone.'))
    selected=read_json(root/'selected_s2/status.json',{})
    folder='selected_s2' if selected.get('selected_backbone') else 'diagnostic_early_sd'
    alignment=read_json(root/folder/'alignment.json',{});status=read_json(root/folder/'status.json',{})
    utility=[]
    for source in ['category','brand','title']:
        vals=[alignment.get(k,{}).get(source) for k in ['mean_delta_ce','mean_delta_rank','validation_ndcg_full_minus_without']]
        utility.append([source]+[f'{v:+.6f}' if v is not None else 'Chưa có' for v in vals])
    tables.append(('5 Căn chỉnh utility và E0 E1',['Nguồn','ΔCE Train-meta','Δrank Train-meta','ΔNDCG Val cùng model'],utility,[1,2,2,2],
                   f"Nhãn: {status.get('label_choice','PENDING')}; E0/E1: {status.get('decision','PENDING')}; phạm vi {'backbone đã chọn' if status.get('selected_backbone') else 'chẩn đoán, không cho phép S3'}. Hai hệ độc lập, cohort eligible-only cho CE; Val giữ OOV miss."))
    pd.DataFrame(utility,columns=['source','delta_ce','delta_rank','delta_ndcg_val']).to_csv(root/'utility_summary.csv',index=False,lineterminator='\n')
    profile=pd.read_csv(root/'dropout_profile.csv') if (root/'dropout_profile.csv').exists() else pd.DataFrame(columns=['subset','part','ndcg_10'])
    rows=[]
    for lab in ['CBT','BT','CT','CB','ID']:
        rows.append([lab]+[summary(profile[(profile.subset==lab)&(profile.part==part)].ndcg_10.to_numpy()) for part in ['val','test']])
    rows.append(['ID riêng',summary(values(jobs('early','ID'),'val')),summary(values(jobs('early','ID')))])
    rows.append(['Early CBT không SD',summary(values(jobs('early','CBT'),'val')),summary(values(jobs('early','CBT')))])
    tables.append(('6 Source dropout',['Chế độ S1','Validation mean ± SD','Test mean ± SD'],rows,[2,2.5,2.5],
                   'ID trong S1 = tắt mọi nguồn, không phải ID-only riêng. Full S1 phải vượt ID, giữ ≥98% early không SD; all-off phải giữ ≥95% ID (validation).'))
    decision=read_json(root/'backbone_decision.json',{});frame=read_json(root/'framing_decision.json',{})
    test_status='Đã có release riêng' if (root/'gate_test.csv').exists() else 'Không áp dụng: chưa có backbone/E1 đủ điều kiện' if (decision and not decision.get('backbone')) or (selected and not selected.get('s3_eligible')) else 'Chưa mở; cần freeze/release-test sau mốc 13/11'
    rows=[['Backbone',str(decision.get('backbone') or decision.get('status','PENDING'))],
          ['Kiểm tra validation',json.dumps(decision.get('checks',{}),ensure_ascii=False)],
          ['Framing',str(frame.get('framing','Chưa chốt'))+'; '+str(frame.get('reason','chỉ dùng validation'))],
          ['UASF Test',test_status],
          ['Tái lập','Số liệu tính từ results.csv/ranks; source, cohort, checkpoint SHA256 được ràng buộc; output cũ giữ nguyên.']]
    tables.append(('7 Đề xuất và trạng thái',['Mục','Nội dung'],rows,[1.5,5.5],None))
    # The final document is rebuilt from this CSV, not from hand-entered values.
    flat=[]
    for i,(title,heads,rows,widths,note) in enumerate(tables):
        for row in rows:flat.append(dict(table=i,title=title,headers=json.dumps(heads,ensure_ascii=False),values=json.dumps(row,ensure_ascii=False),widths=json.dumps(widths),note=note or ''))
    pd.DataFrame(flat).to_csv(root/'report_tables.csv',index=False,lineterminator='\n')
    flat=pd.read_csv(root/'report_tables.csv',keep_default_na=False);body=[_paragraph('Mini checkpoint TAIMMA ngày 07 tháng 11 năm 2026','Title',True,29),_paragraph(root.name+' • mean ± SD qua seed; số liệu còn thiếu được ghi rõ.',size=20)];md=[]
    for i,group in flat.groupby('table',sort=True):
        first=group.iloc[0];heads=json.loads(first.headers);rows=[json.loads(v) for v in group['values']]
        if i==3:body.append('<w:p><w:r><w:br w:type="page"/></w:r></w:p>')
        body.extend([_paragraph(first.title,bold=True,size=22),_table(heads,rows,json.loads(first.widths),first.note or None)])
        md.extend(['## '+first.title,'','| '+' | '.join(heads)+' |','|'+'---|'*len(heads)]+['| '+' | '.join(str(v) for v in row)+' |' for row in rows]+[first.note,''])
        if i==2 and k1:
            outside=k1['train_meta']['mean_ce_late_minus_id']<0
            english=('Metadata improved CE on both Train-base and held-out Train-meta. This does not support an exclusively in-sample fitting explanation. ' if outside and k1['train_base']['mean_ce_late_minus_id']<0 else 'The CE controls describe how residual fitting transfers beyond Train-base. ')
            english+='The validation interventions must be interpreted against their same-seed ID and late references. Joint training combines several changes and does not isolate a unique cause. Single-seed diagnostic results do not establish training-seed robustness.'
            body.append(_paragraph(english,size=19));md.append(english)
    body.append('<w:sectPr><w:pgSz w:w="12240" w:h="15840"/><w:pgMar w:top="650" w:bottom="650" w:left="850" w:right="850"/></w:sectPr>')
    xml=f'<?xml version="1.0" encoding="UTF-8"?><w:document xmlns:w="{NS}"><w:body>'+''.join(body)+'</w:body></w:document>'
    types='<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types"><Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/><Default Extension="xml" ContentType="application/xml"/><Override PartName="/word/document.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/></Types>'
    rels='<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships"><Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="word/document.xml"/></Relationships>'
    target=root/'mini_checkpoint.docx'
    with zipfile.ZipFile(target,'w',zipfile.ZIP_DEFLATED) as z:z.writestr('[Content_Types].xml',types);z.writestr('_rels/.rels',rels);z.writestr('word/document.xml',xml)
    (root/'mini_checkpoint.md').write_text('\n'.join(md),encoding='utf-8')
    return target
