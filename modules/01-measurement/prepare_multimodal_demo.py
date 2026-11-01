# Optional instructor/research recomputation, separate from the beginner default.
# A fresh environment needs requests, pillow, numpy, torch CPU and transformers 4.57.6.
# Original code MIT; external image/data CC0; original model retains its own license.
from pathlib import Path
import base64, hashlib, html, io, json, os, re, time
from concurrent.futures import ThreadPoolExecutor
import requests
from PIL import Image

import argparse, tempfile
parser=argparse.ArgumentParser(description='Recompute the public STATS201 multimodal teaching fixture')
parser.add_argument('--output',type=Path,default=Path(__file__).with_name('multimodal_demo.json'))
args=parser.parse_args()
R=Path(tempfile.mkdtemp(prefix='stats201-demo-'))
A=R/'build/assets'; A.mkdir(parents=True,exist_ok=True)
S=R/'sources/cma'; S.mkdir(parents=True,exist_ok=True)
api='https://openaccess-api.clevelandart.org/api/artworks/?cc0=1&has_image=1&limit=12'
response=requests.get(api,timeout=45); response.raise_for_status()
raw=response.json(); (S/'api_response.json').write_text(json.dumps(raw,ensure_ascii=False,indent=2))
chosen=[x for x in raw['data'] if x.get('share_license_status')=='CC0' and x.get('images',{}).get('web',{}).get('url')][:8]
assert len(chosen)==8
def download(item):
    url=item['images']['web']['url']; r=requests.get(url,timeout=45);r.raise_for_status()
    image=Image.open(io.BytesIO(r.content)).convert('RGB')
    image.save(S/(str(item['id'])+'.png'))
    thumb=image.copy();thumb.thumbnail((300,300));buf=io.BytesIO();thumb.save(buf,format='JPEG',quality=85)
    thumb.save(A/(str(item['id'])+'.jpg'),quality=90)
    text=' '.join(str(item.get(k) or '') for k in ['title','creation_date','technique','culture'])
    text=html.unescape(re.sub('<[^>]+>',' ',text))
    return image,dict(id=item['id'],title=item['title'],date=item.get('creation_date'),technique=item.get('technique'),catalogue_text=text,source_url=item['url'],image_url=url,license='CC0-1.0',download_sha256=hashlib.sha256(r.content).hexdigest(),thumbnail_jpeg=base64.b64encode(buf.getvalue()).decode())
loaded=list(ThreadPoolExecutor(max_workers=4).map(download,chosen))
print('Downloaded 8 observed catalogue/image pairs.',flush=True)
os.environ['HF_HUB_DISABLE_XET']='1'
os.environ['HF_HOME']=str(R/'model_cache')
import torch
from transformers import CLIPModel, CLIPProcessor
torch.set_num_threads(2)
model_id='openai/clip-vit-base-patch32'
meta=requests.get('https://huggingface.co/api/models/'+model_id,timeout=30);meta.raise_for_status();revision='3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268'
print('Loading public original model at',revision,flush=True)
model=CLIPModel.from_pretrained(model_id,revision=revision,token=False).eval()
processor=CLIPProcessor.from_pretrained(model_id,revision=revision,token=False,use_fast=False)
queries=['a portrait of a person','a landscape painting','a photograph of a quantum computer']
with torch.no_grad():
    image_inputs=processor(images=[x[0] for x in loaded],return_tensors='pt')
    image_features=model.get_image_features(**image_inputs)
    image_features=image_features/image_features.norm(dim=-1,keepdim=True)
    text_inputs=processor(text=queries,return_tensors='pt',padding=True)
    text_features=model.get_text_features(**text_inputs)
    text_features=text_features/text_features.norm(dim=-1,keepdim=True)
    catalogue_inputs=processor(text=[x[1]['catalogue_text'] for x in loaded],return_tensors='pt',padding=True,truncation=True)
    catalogue_features=model.get_text_features(**catalogue_inputs)
    catalogue_features=catalogue_features/catalogue_features.norm(dim=-1,keepdim=True)
    scores=(text_features@image_features.T).cpu().numpy()
rows=[]
for i,(_,row) in enumerate(loaded):
    row['image_embedding']=[round(float(v),7) for v in image_features[i]]
    row['catalogue_embedding']=[round(float(v),7) for v in catalogue_features[i]]
    rows.append(row)
bundle=dict(version='2026-11-01',source_api=api,source_policy='https://www.clevelandart.org/open-access',unit='one museum catalogue artwork record joined to its digitized image by artwork ID',n=8,selection='first 8 CC0 records with web images in the saved 12-record API response; convenience teaching sample',frequency='one API retrieval snapshot; artwork creation dates are not sensor observation times',time_range='artwork dates copied individually; retrieved 2026-11-01',model=model_id,model_revision=revision,embedding_dimension=512,preprocessing='CLIPProcessor from the same pinned original model revision; RGB web images; eval; L2 normalization; CPU float32',code='https://github.com/openai/CLIP',computed_by='instructor demonstration executed with the original public model weights; cached encodings, not live inference in the default notebook',rows=rows,queries=[dict(text=q,embedding=[round(float(v),7) for v in text_features[k]],image_scores=[round(float(v),7) for v in scores[k]]) for k,q in enumerate(queries)])
out=args.output;out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(bundle,ensure_ascii=False,separators=(',',':')))
summary={'n':8,'model':model_id,'revision':revision,'dimensions':512,'queries':{q:[{'id':rows[i]['id'],'title':rows[i]['title'],'cosine':round(float(scores[k,i]),4)} for i in scores[k].argsort()[::-1][:3]] for k,q in enumerate(queries)}}
(R/'build/multimodal_results.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2))
print(json.dumps(summary,ensure_ascii=False,indent=2),flush=True)
