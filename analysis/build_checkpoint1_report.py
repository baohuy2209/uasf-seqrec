"""Build the supervisor checkpoint report from finished, measured artifacts."""
import csv
import json
from pathlib import Path
from xml.sax.saxutils import escape
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.enums import TA_LEFT
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import SimpleDocTemplate,Paragraph,Spacer,Table,TableStyle,PageBreak,Image
from pypdf import PdfReader

import argparse
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--run',default='runs/checkpoint1_20261101_batch')
parser.add_argument('--output',default='reports/checkpoint1')
parser.add_argument('--figures',default='.tmp/checkpoint1_figures')
args=parser.parse_args()
ROOT=Path(__file__).resolve().parents[1]
RUN=(ROOT/args.run).resolve()
OUT=(ROOT/args.output).resolve()
OUT.mkdir(parents=True,exist_ok=True)
def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))
def rows(path):return list(csv.DictReader(Path(path).open(encoding='utf-8-sig',newline='')))
progress=read(RUN/'progress.json')
evidence=read(RUN/'e0_e1.json')
assert progress['checkpoint_experiments']=='COMPLETED' and evidence['data_mode']=='real'
font=Path(r'C:\Windows\Fonts\arial.ttf')
font_bold=Path(r'C:\Windows\Fonts\arialbd.ttf')
if not font.exists():
    font=Path('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf')
    font_bold=Path('/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf')
pdfmetrics.registerFont(TTFont('Arial',str(font)))
pdfmetrics.registerFont(TTFont('ArialBold',str(font_bold)))
pdfmetrics.registerFontFamily('Arial',normal='Arial',bold='ArialBold',italic='Arial',boldItalic='ArialBold')
body=ParagraphStyle('Body',fontName='Arial',fontSize=9,leading=12,spaceAfter=5,textColor=colors.HexColor('#20252b'))
small=ParagraphStyle('Small',parent=body,fontSize=7.6,leading=10,spaceAfter=3)
title=ParagraphStyle('Title',parent=body,fontName='ArialBold',fontSize=16,leading=20,spaceAfter=6,textColor=colors.black)
heading=ParagraphStyle('Section',parent=body,fontName='ArialBold',fontSize=10,leading=13,spaceBefore=7,spaceAfter=4)
story=[]
def p(text,style=body):return Paragraph(text,style)
def add(text,style=body):story.append(p(text,style))
def section(text):story.append(p(text,heading))
def table(data,widths):
    cells=[[p(escape(str(v)),small) for v in row] for row in data]
    value=Table(cells,colWidths=widths,hAlign='LEFT',repeatRows=1)
    value.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.HexColor('#eaf0f5')),
          ('VALIGN',(0,0),(-1,-1),'TOP'),('LINEBELOW',(0,0),(-1,0),.5,colors.HexColor('#9daab7')),
          ('BOTTOMPADDING',(0,0),(-1,-1),3),('TOPPADDING',(0,0),(-1,-1),3),
          ('LEFTPADDING',(0,0),(-1,-1),4),('RIGHTPADDING',(0,0),(-1,-1),4)]))
    story.append(value)
def f(value,digits=4):return f'{float(value):.{digits}f}' if value is not None else 'Không xác định'

story.append(p('Báo cáo Checkpoint 1 TAIMMA',title))
hardware=read(Path(progress['id_seed42']['folder'])/'history.json')['resource_usage'].get('gpu_name') or 'CPU'
add('Beauty và Amazon 2014/5-core | Kết quả chạy thật trên '+escape(hardware))
add(f'<b>Đề xuất {escape(evidence["decision"])}</b>. Kết luận dựa trên nhãn utility từ hai hệ độc lập, E0/E1 và quy tắc đã khóa; không dựa vào dữ liệu giả lập.')
section('Pilot và giao thức đánh giá')
loo=progress[f'loo_d{progress["backbone_dim"]}']
add(f'Điều kiện (3): <b>PASS</b>; LOO chuẩn, seed 42, d = {progress["backbone_dim"]}; test Recall@10 = {f(loo["test"]["Recall@10"])}, '
    f'NDCG@10 = {f(loo["test"]["NDCG@10"])} (ngưỡng 0,075 và 0,033). Checkpoint chọn bằng validation; giữ cấu hình này cho backbone.')
add('Thí nghiệm chính dùng <b>input_history_strict</b>: mask mọi ID có trong chuỗi đầu vào sau tính điểm; không miễn target. '
    'LOO đối chiếu riêng dùng không mask history, khớp loader sequential của DIF-SR. Không so trực tiếp hai bảng như cùng giao thức.')
audit=progress['audit']
add(f'Beauty: 109.050 cặp Train-base; LOO: 131.413. Target trùng input: validation {audit["validation"]["overlap"]}, test {audit["test"]["overlap"]}. '
    f'Target OOV: validation {audit["validation"]["oov"]}, test {audit["test"]["oov"]} trên 22.363 lượt mỗi split; đều tính là trượt. '
    'Lặp seed 42 đã đối chiếu từng rank. Mỗi run lưu checkpoint, config, git diff và snapshot mã.')
section('Bảng 1 Dữ liệu và độ phủ metadata')
data=[['Tập','Users','Items','Tương tác','Cat %','Brand %','Title %']]
for key in ['beauty','sports','toys']:
    directory=ROOT/'artifacts/metadata'/key
    cov=read(directory/'coverage.json');stats=read(directory/'statistics.json');scope=cov['all_unique_items_cohort']
    data.append([key.title(),f'{stats["users"]:,}',f'{stats["original_items"]:,}',f'{stats["interactions"]:,}',
                 f(scope['category']['coverage_pct'],2),f(scope['brand']['coverage_pct'],2),f(scope['title']['coverage_pct'],2)])
table(data,[74,68,68,90,68,68,68])
add('Độ phủ tính trên toàn bộ item cohort. Catalog Train-base lần lượt 11.998 / 18.132 / 11.766 item. Category giữ nhiều nhãn bỏ root; Title là MiniLM 384 chiều thật. Mask tự nhiên giữ nguyên khi hoán vị.',)
section('Chẩn đoán B Hợp nhất muộn')
summary=rows(RUN/'diagnostic_B_summary.csv')
label={'ID':'ID','category':'C','brand':'B','title':'T','category+brand':'CB','category+title':'CT','brand+title':'BT','category+brand+title':'CBT'}
data=[['Nguồn','NDCG@10 mean ± SD','Nguồn','NDCG@10 mean ± SD']]
indexed={label[r['sources']]:r for r in summary}
for a,b in [('ID','CB'),('C','CT'),('B','BT'),('T','CBT')]:
    data.append([a,f'{f(indexed[a]["mean"])} ± {f(indexed[a]["std"])}',b,f'{f(indexed[b]["mean"])} ± {f(indexed[b]["std"])}'])
table(data,[55,217,55,227])
for source in ['category','brand','title']:
    values=[r for seed in [42,43,44] for r in rows(RUN/f'marginal_utility_seed{seed}.csv') if r['source_name']==source]
    add(f'{source.title()}: lợi ích thêm riêng = {f(sum(float(r["add_m"]) for r in values)/3)}; '
        f'leave-one-out = {f(sum(float(r["drop_m"]) for r in values)/3)}.',small)

story.append(PageBreak())
story.append(p('Độ bền nhãn utility và quyết định',title))
section('Thiếu metadata và hoán vị khi suy luận')
figure=(ROOT/args.figures).resolve()/'missingness.png'
story.append(Image(str(figure),width=554,height=170))
add('Mỗi đường làm thiếu riêng một nguồn từ 0-80% số item vốn có nguồn đó; các nguồn khác giữ nguyên. Dải màu là ±1 SD trên 3 seed. '
    'Cùng mặt nạ được áp dụng cho lịch sử và ứng viên; không cập nhật trọng số ở các lượt suy luận.',small)
control=[r for r in rows(RUN/'robustness.csv') if r['control']=='permutation']
add('Hoán vị khi suy luận, NDCG@10 trung bình: '+ '; '.join(f'{s.title()} = {f(sum(float(r["NDCG@10"]) for r in control if r["source"]==s)/3)}' for s in ['category','brand','title'])+'.')
section('Nhãn utility và E0 E1')
add('Hai hệ tham chiếu seed 42/43 gồm ID và các nhánh phụ độc lập; Δ = CE(without source) - CE(all) trên Train-meta in-catalog. '
    'ε là median |Δ42 - Δ43|; bootstrap theo user, 1.000 lượt; E1 dùng OOF theo user và chỉ feature lịch sử được phép.')
data=[['Nguồn','ε','Spearman ρ','E0','AUC LR [95% CI]','E1']]
for source,result in evidence['sources'].items():
    stability,e0,e1=result['stability'],result['e0'],result['e1'];ci=e1['bootstrap_95_ci']
    auc=f'{f(e1["oof_auc_lr"],3)} [{f(ci[0],3)}, {f(ci[1],3)}]' if ci else 'Không xác định'
    e0_text=('Đạt' if e0['e0_passed'] else 'Không đạt') + f'; +{100*e0["frac_positive"]:.1f}% / -{100*e0["frac_negative"]:.1f}%. '
    e0_text+=f'Q1 [{f(e0["q1_ci"][0])}; {f(e0["q1_ci"][1])}], Q5 [{f(e0["q5_ci"][0])}; {f(e0["q5_ci"][1])}]'
    data.append([source.title(),f(stability['epsilon']),f(stability['spearman_rho'],3),
                 e0_text,auc,'Đạt' if e1['e1_passed'] else ('Không đạt' if e1.get('oof_auc_lr') is not None else 'Không xác định')])
table(data,[60,48,50,190,145,61])
distribution=rows(RUN/'delta_distribution.csv')
add('Phân phối Δ, trung vị [5%; 95%]: '+ '; '.join(f'{r["source_name"]}: {f(r["50%"])} [{f(r["5%"])}; {f(r["95%"])}]' for r in distribution)+'.',small)
add('Quy tắc: E0 cần ≥15% Δ > ε và ≥15% Δ < -ε hoặc Q1/Q5 trái dấu với CI loại 0. E1 cần AUC ≥0,55, cận dưới CI >0,5 và hơn length-only. '
    'ρ ≥0,3 cho contextual; 0,1-0,3 cho nhãn trung bình/soft; dưới 0,1 cho group-level. Không điều chỉnh ngưỡng sau khi xem kết quả.',small)
section('Chi phí và hành động tiếp theo')
entries=[e for e in progress.values() if isinstance(e,dict) and 'training_sec' in e]
hours=sum(e['training_sec'] for e in entries)/3600
throughput=read(RUN/'throughput.json')
unit='GPU-job giờ' if read(RUN/'protocol.lock.json')['device']=='cuda' else 'CPU-job giờ'
add(f'Tổng thời gian huấn luyện đã đo: {hours:.2f} {unit} (cộng từng job, có thể chồng lấp khi chạy song song). '
    f'Benchmark một epoch dữ liệu thật, thông lượng tốt nhất ở {throughput["best_workers"]} tiến trình; phép đo loại thời gian load/validation.')
budget=read(RUN/'budget_projection.json')
if budget['remaining_estimated_gpu_job_hours'] is not None:
    add(f'Ước lượng phần còn lại cho tuần 03-09/11: {budget["remaining_estimated_gpu_job_hours"]:.1f} GPU-job giờ; '
        'ngoại suy Sports ×1,8 và Toys ×0,8 từ Beauty, cộng khối gate/baseline chưa đo theo dự toán của thầy. Bảng chi tiết tách số đo và ước lượng.',small)
else:
    add('Chưa có phép đo GPU phù hợp để ngoại suy GPU-giờ; không chuyển thời gian CPU thành thời gian GPU.',small)
add(f'<b>Quyết định {escape(evidence["decision"])}</b>; nguồn đáp ứng: {escape(str(evidence["qualifying_sources"]))}; chế độ gate: {escape(str(evidence["gating_regime"]))}. '
    'Chỉ mở rộng sang giai đoạn tiếp theo theo quyết định này. Sports/Toys và các phép thử chưa chạy phải giữ là ước lượng, không ghi thành kết quả đo.')
add('Bằng chứng: runs/checkpoint1_20261101; protocol.lock.json, progress.json, source_snapshot.zip, results.csv, utility.csv và sidecar, e0_e1.json. '
    'Tham chiếu masking: github.com/AIM-SE/DIF-SR, recbole/data/dataloader/general_dataloader.py và recbole/trainer/trainer.py.',small)
output=OUT/'Bao_cao_Checkpoint_1_TAIMMA.pdf'
def footer(canvas,doc):
    canvas.setFont('Arial',7);canvas.setFillColor(colors.HexColor('#69737d'))
    canvas.drawString(22,18,'TAIMMA | Kết quả thực nghiệm và giao thức đã khóa')
    canvas.drawRightString(A4[0]-22,18,str(doc.page))
doc=SimpleDocTemplate(str(output),pagesize=A4,leftMargin=21,rightMargin=21,topMargin=23,bottomMargin=28)
doc.build(story,onFirstPage=footer,onLaterPages=footer)
count=len(PdfReader(str(output)).pages)
assert count==2,f'Report exceeded page limit: {count}'
print(output)
