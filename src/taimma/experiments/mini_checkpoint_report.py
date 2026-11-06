"""Two-page, table-only mini-checkpoint DOCX using the Python standard library.

No Office/python-docx installation or environment upgrade is required on the
RTX4060 host. Missing outputs remain explicitly pending, never filled with zero.
"""
from pathlib import Path
from statistics import mean,stdev
from xml.sax.saxutils import escape
import json
import zipfile

NS='http://schemas.openxmlformats.org/wordprocessingml/2006/main'


def summary(rows):
    values={int(r['seed']):float(r['value']) for r in rows}
    if set(values)!={42,43,44}:return f'{len(values)}/3 seed' if values else 'Chưa có'
    v=[values[s] for s in [42,43,44]]
    return f'{mean(v):.6f} ± {stdev(v):.6f}'


def _paragraph(text,style=None,bold=False,size=22):
    prop=f'<w:pStyle w:val="{style}"/>' if style else ''
    return (f'<w:p><w:pPr>{prop}<w:spacing w:before="0" w:after="0" w:line="230" w:lineRule="auto"/></w:pPr>'
        f'<w:r><w:rPr><w:rFonts w:ascii="Times New Roman" w:hAnsi="Times New Roman"/>'
        f'<w:sz w:val="{size}"/>{"<w:b/>" if bold else ""}<w:color w:val="000000"/></w:rPr>'
        f'<w:t xml:space="preserve">{escape(str(text))}</w:t></w:r></w:p>')


def _table(headers,rows,widths,note=None):
    widths=[int(10440*w/sum(widths)) for w in widths]
    borders=''.join(f'<w:{e} w:val="single" w:sz="4" w:color="D9D9D9"/>' for e in ['top','left','bottom','right','insideH','insideV'])
    grid=''.join(f'<w:gridCol w:w="{w}"/>' for w in widths)
    out=[f'<w:tbl><w:tblPr><w:tblW w:w="10440" w:type="dxa"/><w:tblLayout w:type="fixed"/><w:tblBorders>{borders}</w:tblBorders>'
        '<w:tblCellMar><w:top w:w="45" w:type="dxa"/><w:bottom w:w="45" w:type="dxa"/><w:left w:w="75" w:type="dxa"/><w:right w:w="75" w:type="dxa"/></w:tblCellMar></w:tblPr>'
        f'<w:tblGrid>{grid}</w:tblGrid>']
    for ri,row in enumerate([headers]+list(rows)):
        out.append('<w:tr><w:trPr><w:cantSplit/>'+('<w:tblHeader/>' if ri==0 else '')+'</w:trPr>')
        for ci,text in enumerate(row):
            shading='<w:shd w:fill="E7EDF2"/>' if ri==0 else ''
            out.append(f'<w:tc><w:tcPr><w:tcW w:w="{widths[ci]}" w:type="dxa"/>{shading}<w:vAlign w:val="center"/></w:tcPr>'+_paragraph(text,bold=ri==0)+'</w:tc>')
        out.append('</w:tr>')
    if note:
        out.append(f'<w:tr><w:trPr><w:cantSplit/></w:trPr><w:tc><w:tcPr><w:gridSpan w:val="{len(headers)}"/>'
            '<w:tcW w:w="10440" w:type="dxa"/></w:tcPr>'+_paragraph(note,size=20)+'</w:tc></w:tr>')
    out.append('</w:tbl>')
    return ''.join(out)


def report_tables(state):
    early=state.get('early',{});tables=[]
    tables.append(('1 Tích hợp sớm B',['Nguồn','Seed 42','Seed 43','Seed 44','Mean ± SD'],[
        [label]+[f'{next((float(r["value"]) for r in early.get(key,[]) if r["seed"]==s),float("nan")):.6f}'
            if any(r['seed']==s for r in early.get(key,[])) else 'Chưa có' for s in [42,43,44]]+[summary(early.get(key,[]))]
        for key,label in [('ID','∅ ID riêng'),('C','C'),('B','B'),('T','T'),('CB','CB'),('CT','CT'),('BT','BT'),('CBT','CBT')]
        ],[1.,1.05,1.05,1.05,2.95],'C/B/T = Category/Brand/Title. SD mẫu qua 3 seed; không suy SD từ một seed.'))
    marginal=[]
    for src,key,without in [('Category','C','BT'),('Brand','B','CT'),('Title','T','CB')]:
        def delta(a,b):
            x={r['seed']:r['value'] for r in early.get(a,[])};y={r['seed']:r['value'] for r in early.get(b,[])}
            return summary([dict(seed=s,value=x[s]-y[s]) for s in [42,43,44] if s in x and s in y])
        marginal.append([src,delta(key,'ID'),delta('CBT',without)])
    tables.append(('', ['Nguồn early fusion','Add = nguồn − ID','Drop = CBT − bỏ nguồn'],marginal,[2.5,2.3,2.3],None))
    controls=[]
    for src,label in [('category','Category'),('brand','Brand'),('title','Title')]:
        def ci(kind):
            rec=state.get('controls',{}).get(src+'_'+kind)
            if not rec:return 'Chưa có'
            return summary(rec.get('control_by_seed',[]))+f'; Δ {rec["delta_control_minus_real"]:+.5f} [{rec["ci95"][0]:+.5f}, {rec["ci95"][1]:+.5f}]'
        controls.append([label,summary(early.get('CBT',[])),ci('shuffled'),ci('random')])
    tables.append(('2 Đối chứng khi huấn luyện trên early fusion',['Nguồn','Thật CBT','Hoán vị Δ [CI95]','Ngẫu nhiên Δ [CI95]'],controls,
        [1.,1.9,2.1,2.1],'Δ = đối chứng − thật; bootstrap ghép cặp theo người dùng, giữ chung dự đoán 3 seed; không phải CI biến thiên seed.'))
    diagnostics=state.get('diagnostics',{});diag=[]
    for k in ['K1','K2','K3','K4']:
        rec=diagnostics.get(k)
        if rec and k=='K1':
            text=f'CE late−ID: base {rec["train_base"]["mean_ce_late_minus_id"]:+.4f}; meta {rec["train_meta"]["mean_ce_late_minus_id"]:+.4f}'
        elif rec:text=f'Val {rec["val_ndcg_10"]:.6f}; Test {rec["test_ndcg_10"]:.6f}; {rec["stop_reason"]}'
        else:text='Chưa có kết quả đối chứng'
        diag.append([k,text])
    conclusion='Chưa đủ K1–K4 để chốt nguyên nhân.'
    if len(diagnostics)==4:
        k1=diagnostics['K1'];supported=k1['train_base']['mean_ce_late_minus_id']<0 and k1['train_meta']['mean_ce_late_minus_id']>=0
        conclusion=('K1 phù hợp giả thuyết học phần dư trong mẫu; ' if supported else 'K1 chưa xác nhận mẫu hình học phần dư trong mẫu; ')+ 'K2–K4 mô tả các đối chứng dữ liệu, điều chuẩn và gói huấn luyện chung.'
    tables.append(('3 Nguyên nhân hợp nhất muộn',['Mã','Kết quả K1 đến K4'],diag,[.55,6.55],conclusion))
    groups=[]
    for mech,label in [('late_legacy','Late fusion'),('early','Early fusion')]:
        rows=state.get('groups',{}).get(mech,[])
        def gsummary(field):return summary([dict(seed=r['seed'],value=r[field]) for r in rows])
        subsets=', '.join(sorted({r['global_subset'] for r in rows}))
        groups.append([label,gsummary('id_test'),(subsets+'; ' if subsets else '')+gsummary('global_test'),
            gsummary('group_val_selected_test'),gsummary('group_test_oracle')])
    tables.append(('4 Lựa chọn theo nhóm',['Cơ chế','ID riêng','Subset toàn cục','Nhóm Val → Test','Oracle nhóm Test'],groups,[.8,1.5,1.8,1.5,1.5],
        'Ranh giới nhóm fit trên lịch sử validation và giữ nguyên khi áp dụng test; oracle dùng nhãn test chỉ là trần mô tả.'))
    utility=state.get('utility',{});rows=[]
    for s,label in [('category','Category'),('brand','Brand'),('title','Title')]:
        vals=[]
        for field in ['mean_delta_ce','mean_delta_rank','validation_ndcg_full_minus_without']:
            v=utility.get(field,{}).get(s);vals.append(f'{v:+.6f}' if v is not None else 'Chưa có')
        rows.append([label]+vals)
    tables.append(('5 Nhãn utility',['Nguồn','Mean ΔCE','Mean Δrank','ΔNDCG Val cùng model'],rows,[1.,1.5,1.5,3.1],
        'Nhãn chọn: '+utility.get('label_choice','PENDING')+'; CE nếu khớp dấu cả ba nguồn, nếu không xét log-rank; không chọn bằng test.'))
    dropout=state.get('source_dropout',{})
    rows=[[label,summary(dropout.get(key,[]))] for key,label in [('CBT','Đủ C B T'),('BT','Tắt Category'),('CT','Tắt Brand'),('CB','Tắt Title'),('ID','Tắt hết trong S1')]]
    rows.append(['ID-only riêng',summary(early.get('ID',[]))])
    tables.append(('6 Early fusion có source dropout',['Chế độ','NDCG@10 mean ± SD'],rows,[3.7,3.4],
        'Tắt hết là chế độ của S1 đã học, không phải checkpoint ID-only riêng. Cùng mask nguồn ở lịch sử và ứng viên.'))
    proposal=state.get('proposal','Chưa chốt')
    if isinstance(proposal,dict):proposal=f'{proposal["backbone"]}: {proposal["reason"]}'
    if len(proposal)>240:proposal=proposal[:237]+'...'
    tables.append(('7 Đề xuất kiến trúc nền cho UASF',['Mục','Đề xuất và lý do'],[
        ['Kiến trúc nền',proposal],['Trạng thái','Đủ kết quả mini-checkpoint' if state.get('complete') else 'Chưa đủ kết quả; các ô thiếu cần hoàn tất trước khi chốt'],
        ['S2 và S3','Chọn nền bằng validation; sinh lại S2/E1 đầy đủ ψ; chỉ chạy S3 khi E0/E1 đạt đúng chế độ. Không tự chạy S3 trên late fusion cũ.']],
        [1.55,5.55],None))
    return tables


def write_mini_report(state,output_dir):
    output_dir=Path(output_dir);output_dir.mkdir(parents=True,exist_ok=True)
    (output_dir/'mini_checkpoint.json').write_text(json.dumps(state,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8',newline='\n')
    tables=report_tables(state)
    body=[_paragraph('Mini checkpoint ngày 06 tháng 11 năm 2026','Title',True,32)]
    body.append(_table(['Phạm vi','Trạng thái dữ liệu'],[[state.get('dataset','beauty')+' • test NDCG@10',
        'Đủ kết quả đã kiểm chứng' if state.get('complete') else 'Báo cáo tiến độ; chưa đủ output để chốt. Chưa có không có nghĩa bằng 0.']],[1.55,5.55]))
    markdown=['# Mini checkpoint 06/11/2026','']
    for index,(title,headers,rows,widths,note) in enumerate(tables):
        if index==4:body.append('<w:p><w:r><w:br w:type="page"/></w:r></w:p>')
        if title:
            body.append('<w:p><w:pPr><w:keepNext/><w:spacing w:before="80" w:after="40"/></w:pPr><w:r><w:rPr><w:b/><w:sz w:val="23"/></w:rPr><w:t>'+escape(title)+'</w:t></w:r></w:p>')
            markdown.extend(['**'+title+'**',''])
        body.append(_table(headers,rows,widths,note))
        markdown.extend(['| '+' | '.join(headers)+' |','|'+'---|'*len(headers)])
        markdown.extend('| '+' | '.join(str(x).replace('|','/') for x in row)+' |' for row in rows)
        if note:markdown.append('| '+note+' |'+' |'*(len(headers)-1))
        markdown.append('')
    body.append('<w:sectPr><w:footerReference w:type="default" r:id="rIdFooter"/><w:pgSz w:w="12240" w:h="15840"/><w:pgMar w:top="720" w:bottom="720" w:left="900" w:right="900" w:footer="360"/></w:sectPr>')
    doc=f'<?xml version="1.0" encoding="UTF-8" standalone="yes"?><w:document xmlns:w="{NS}" xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships"><w:body>'+''.join(body)+'</w:body></w:document>'
    styles=f'<w:styles xmlns:w="{NS}"><w:docDefaults><w:rPrDefault><w:rPr><w:rFonts w:ascii="Times New Roman" w:hAnsi="Times New Roman"/><w:sz w:val="22"/><w:color w:val="000000"/></w:rPr></w:rPrDefault><w:pPrDefault><w:pPr><w:spacing w:after="0" w:line="230" w:lineRule="auto"/></w:pPr></w:pPrDefault></w:docDefaults><w:style w:type="paragraph" w:default="1" w:styleId="Normal"><w:name w:val="Normal"/></w:style><w:style w:type="paragraph" w:styleId="Title"><w:name w:val="Title"/><w:basedOn w:val="Normal"/><w:pPr><w:spacing w:after="100"/></w:pPr></w:style></w:styles>'
    footer=f'<w:ftr xmlns:w="{NS}"><w:p><w:pPr><w:jc w:val="right"/></w:pPr><w:r><w:t>Trang </w:t></w:r><w:fldSimple w:instr="PAGE"/></w:p></w:ftr>'
    types='<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types"><Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/><Default Extension="xml" ContentType="application/xml"/><Override PartName="/word/document.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/><Override PartName="/word/styles.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.styles+xml"/><Override PartName="/word/footer.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.footer+xml"/></Types>'
    rels='<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships"><Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="word/document.xml"/></Relationships>'
    docrels='<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships"><Relationship Id="rIdStyles" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/styles" Target="styles.xml"/><Relationship Id="rIdFooter" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/footer" Target="footer.xml"/></Relationships>'
    path=output_dir/'mini_checkpoint_0611.docx'
    with zipfile.ZipFile(path,'w',compression=zipfile.ZIP_DEFLATED) as z:
        for name,data in {'[Content_Types].xml':types,'_rels/.rels':rels,'word/document.xml':doc,
            'word/styles.xml':styles,'word/footer.xml':footer,'word/_rels/document.xml.rels':docrels}.items():z.writestr(name,data)
    (output_dir/'mini_checkpoint.md').write_text('\n'.join(markdown),encoding='utf-8',newline='\n')
    return path
