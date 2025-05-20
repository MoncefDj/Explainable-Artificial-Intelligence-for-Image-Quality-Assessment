# app.py
from flask import Flask, render_template, request, jsonify
import os
import copy
import traceback
import cv2
import torch
import numpy as np
import pandas as pd
import albumentations as A
import torch.nn as nn
from torch.utils.data import Dataset 
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import io
from PIL import Image
import base64
import math
import time

from pyngrok import ngrok, conf 

GPT4ALL_AVAILABLE = False
try:
    from gpt4all import GPT4All
    GPT4ALL_AVAILABLE = True
except ImportError:
    print("Flask Warning: gpt4all library not found. LLM explanation will be skipped.")
    GPT4All = None

from segmentation_models_pytorch import DeepLabV3Plus

# --- Constants ---
IMG_SIZE = 256; NORMGRAD_EPSILON = 0.0005; IMG_SIZE_ANALYSIS = IMG_SIZE
NORMGRAD_EPSILON_ANALYSIS = NORMGRAD_EPSILON; DEFAULT_SIGMOID_K = 15.0; DEFAULT_SIGMOID_THRESH = 0.005
DATA_BASE_PATH = "/home/linati/SegmToClass/Train/object-CXR" 
TRAINED_MODEL_PATH = "/home/linati/SegmToClass/Train/loss_experiment_outputs/focal_tversky_loss/focal_tversky_loss_best_model.pth" 
TEST_CSV_NAME = "dev.csv"
NORMGRAD_TARGET_LAYERS_LIST = [
    'encoder.conv1', 'encoder.maxpool', 'encoder.layer1', 'encoder.layer1.0', 'encoder.layer1.0.conv1',
    'encoder.layer1.0.conv2', 'encoder.layer1.0.conv3', 'encoder.layer1.0.downsample', 'encoder.layer1.0.downsample.0',
    'encoder.layer1.0.downsample.1', 'encoder.layer1.1', 'encoder.layer1.1.conv1', 'encoder.layer1.1.conv2',
    'encoder.layer1.1.conv3', 'encoder.layer1.2', 'encoder.layer1.2.conv1', 'encoder.layer1.2.conv2',
    'encoder.layer1.2.conv3', 'encoder.layer2', 'encoder.layer2.0', 'encoder.layer2.0.conv1', 'encoder.layer2.0.conv2',
    'encoder.layer2.0.conv3', 'encoder.layer2.0.downsample', 'encoder.layer2.0.downsample.0', 'encoder.layer2.0.downsample.1',
    'encoder.layer2.1', 'encoder.layer2.1.conv1', 'encoder.layer2.1.conv2', 'encoder.layer2.1.conv3', 'encoder.layer2.2',
    'encoder.layer2.2.conv1', 'encoder.layer2.2.conv2', 'encoder.layer2.2.conv3', 'encoder.layer2.3',
    'encoder.layer2.3.conv1', 'encoder.layer2.3.conv2', 'encoder.layer2.3.conv3', 'encoder.layer3', 'encoder.layer3.0',
    'encoder.layer3.0.conv1', 'encoder.layer3.0.conv2', 'encoder.layer3.0.conv3', 'encoder.layer3.0.downsample',
    'encoder.layer3.0.downsample.0', 'encoder.layer3.0.downsample.1', 'encoder.layer3.1', 'encoder.layer3.1.conv1',
    'encoder.layer3.1.conv2', 'encoder.layer3.1.conv3', 'encoder.layer3.2', 'encoder.layer3.2.conv1',
    'encoder.layer3.2.conv2', 'encoder.layer3.2.conv3', 'encoder.layer3.3', 'encoder.layer3.3.conv1',
    'encoder.layer3.3.conv2', 'encoder.layer3.3.conv3', 'encoder.layer3.4', 'encoder.layer3.4.conv1',
    'encoder.layer3.4.conv2', 'encoder.layer3.4.conv3', 'encoder.layer3.5', 'encoder.layer3.5.conv1',
    'encoder.layer3.5.conv2', 'encoder.layer3.5.conv3', 'encoder.layer4', 'encoder.layer4.0', 'encoder.layer4.0.conv1',
    'encoder.layer4.0.conv2', 'encoder.layer4.0.conv3', 'encoder.layer4.0.downsample', 'encoder.layer4.0.downsample.0',
    'encoder.layer4.0.downsample.1', 'encoder.layer4.1', 'encoder.layer4.1.conv1', 'encoder.layer4.1.conv2',
    'encoder.layer4.1.conv3', 'encoder.layer4.2', 'encoder.layer4.2.conv1', 'encoder.layer4.2.conv2',
    'encoder.layer4.2.conv3', 'decoder', 'decoder.aspp', 'decoder.aspp.0', 'decoder.aspp.0.convs.0',
    'decoder.aspp.0.convs.0.0', 'decoder.aspp.0.convs.0.1', 'decoder.aspp.0.convs.0.2', 'decoder.aspp.0.convs.1',
    'decoder.aspp.0.convs.1.0', 'decoder.aspp.0.convs.1.0.0', 'decoder.aspp.0.convs.1.0.1', 'decoder.aspp.0.convs.1.1',
    'decoder.aspp.0.convs.1.2', 'decoder.aspp.0.convs.2', 'decoder.aspp.0.convs.2.0', 'decoder.aspp.0.convs.2.0.0',
    'decoder.aspp.0.convs.2.0.1', 'decoder.aspp.0.convs.2.1', 'decoder.aspp.0.convs.2.2', 'decoder.aspp.0.convs.3',
    'decoder.aspp.0.convs.3.0', 'decoder.aspp.0.convs.3.0.0', 'decoder.aspp.0.convs.3.0.1', 'decoder.aspp.0.convs.3.1',
    'decoder.aspp.0.convs.3.2', 'decoder.aspp.0.convs.4', 'decoder.aspp.0.convs.4.0', 'decoder.aspp.0.convs.4.1',
    'decoder.aspp.0.convs.4.2', 'decoder.aspp.0.convs.4.3', 'decoder.aspp.0.project', 'decoder.aspp.0.project.0',
    'decoder.aspp.0.project.1', 'decoder.aspp.0.project.2', 'decoder.aspp.0.project.3', 'decoder.aspp.1',
    'decoder.aspp.1.0', 'decoder.aspp.1.1', 'decoder.aspp.2', 'decoder.aspp.3', 'decoder.up', 'decoder.block1',
    'decoder.block1.0', 'decoder.block1.1', 'decoder.block1.2', 'decoder.block2', 'decoder.block2.0',
    'decoder.block2.0.0', 'decoder.block2.0.1', 'decoder.block2.1', 'decoder.block2.2', 'segmentation_head',
    'segmentation_head.0', 'segmentation_head.1', 'segmentation_head.2', 'segmentation_head.2.activation'
]
analysis_config_global = {
    "USE_LLM_EXPLANATION": True,
    "LLM_MODEL_NAME": "Meta-Llama-3-8B-Instruct.Q4_0.gguf",
    "LLM_DEVICE": "cuda" if torch.cuda.is_available() else "cpu",
    "WEIGHT_IMPORTANCE": 0.3, "WEIGHT_SIZE_PENALTY": 0.7,
    "SIGMOID_K_PARAM": 70.0, "SIGMOID_THRESH_PARAM": 0.1, "SALIENCY_FILTER_THRESHOLD": 0.30,
    "NORMGRAD_EPSILON": NORMGRAD_EPSILON_ANALYSIS, "NORMGRAD_ADVERSARIAL": False
}

# --- Helper Functions (Dataset, Model, Analysis Logic) ---
def parse_annotations(ann_str, img_shape):
    mask = np.zeros(img_shape[:2], dtype=np.uint8)
    if not isinstance(ann_str, str) or ann_str.strip() == "": return mask
    h, w = img_shape[:2]
    try:
        objects = ann_str.split(';')
        for obj_item in objects: 
            if not obj_item.strip(): continue
            parts = list(map(int, obj_item.split()))
            obj_type, coords = parts[0], parts[1:]
            if obj_type == 0 and len(coords) == 4: 
                x1,y1,x2,y2 = map(lambda v:max(0,v), coords); x2,y2=min(x2,w),min(y2,h)
                if x1<x2 and y1<y2: mask[y1:y2,x1:x2]=1
            elif obj_type == 1 and len(coords) == 4: 
                x1,y1,x2,y2 = map(lambda v:max(0,v), coords); x2,y2=min(x2,w),min(y2,h)
                if x1<x2 and y1<y2:
                    center,axes = ((x1+x2)//2,(y1+y2)//2),((x2-x1)//2,(y2-y1)//2)
                    if axes[0]>0 and axes[1]>0: cv2.ellipse(mask,center,axes,0,0,360,1,-1)
                    elif axes[0]==0 and axes[1]>0: cv2.line(mask,(center[0],y1),(center[0],y2),1,1) 
                    elif axes[1]==0 and axes[0]>0: cv2.line(mask,(x1,center[1]),(x2,center[1]),1,1)
            elif obj_type == 2 and len(coords)>=6 and len(coords)%2==0: 
                points=np.array(coords).reshape(-1,2)
                points[:,0]=np.clip(points[:,0],0,w-1); points[:,1]=np.clip(points[:,1],0,h-1)
                cv2.fillPoly(mask,[points],1)
    except Exception as e: print(f"Flask: Error parsing annotation: {e}")
    return mask

class ObjectCXRSataset(Dataset):
    def __init__(self, csv_file, img_dir, transform=None, img_size=(IMG_SIZE,IMG_SIZE)):
        self.data = pd.DataFrame()
        try:
            self.data = pd.read_csv(csv_file)
            if self.data.empty: print(f"Flask Warning: CSV {csv_file} is empty.")
        except Exception as e:
            print(f"Flask Error reading CSV {csv_file}: {e}"); self.data = pd.DataFrame()
        self.img_dir, self.transform, self.img_size = img_dir, transform, img_size
    def __len__(self): return len(self.data) if hasattr(self.data, 'empty') and not self.data.empty else 0
    def __getitem__(self, idx):
        if not (0 <= idx < len(self.data)):
            print(f"Flask Critical Error: Index {idx} out of range for {len(self.data)}")
            return torch.zeros((3,*self.img_size),dtype=torch.float32), torch.zeros((1,*self.img_size),dtype=torch.float32)
        img_name, ann_str = self.data.iloc[idx,0], self.data.iloc[idx,1] if self.data.shape[1]>1 else ""
        img_path = os.path.join(self.img_dir,img_name)
        try:
            img = cv2.imread(img_path)
            if img is None: print(f"Flask Warning: Fail load {img_path}. Dummy."); return torch.zeros((3,*self.img_size),dtype=torch.float32), torch.zeros((1,*self.img_size),dtype=torch.float32)
            img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB); mask_data=parse_annotations(ann_str,img.shape)
            if self.transform: aug=self.transform(image=img,mask=mask_data); img,mask_data=aug['image'],aug['mask']
            else: img = cv2.resize(img,self.img_size,interpolation=cv2.INTER_AREA); mask_data = cv2.resize(mask_data,self.img_size,interpolation=cv2.INTER_NEAREST)
            if img.dtype==np.uint8: img=img/255.0
            return torch.tensor(img,dtype=torch.float32).permute(2,0,1), torch.tensor(mask_data,dtype=torch.float32).unsqueeze(0)
        except Exception as e: print(f"Flask Error item {idx} ({img_name}): {e}"); traceback.print_exc(); return torch.zeros((3,*self.img_size),dtype=torch.float32),torch.zeros((1,*self.img_size),dtype=torch.float32)

test_transform_global = A.Compose([A.Resize(IMG_SIZE,IMG_SIZE)])
class SegmentationModel(nn.Module):
    def __init__(self,enc="resnet50",w="imagenet",cls=1): super().__init__(); self.model=DeepLabV3Plus(encoder_name=enc,encoder_weights=w,classes=cls,activation=None)
    def forward(self,x): return self.model(x)

def find_layer_analysis(model_s,layer_n):
    s_base,c_path = model_s,layer_n
    if hasattr(model_s,'model') and isinstance(model_s.model,nn.Module) and not layer_n.startswith('model.'): s_base=model_s.model
    elif layer_n.startswith('model.'): c_path=layer_n[len('model.'):]
    c_mod=s_base
    try:
        for p_item in c_path.split('.'):
            if hasattr(c_mod,p_item): c_mod=getattr(c_mod,p_item)
            else:
                try: idx=int(p_item)
                except ValueError: return None
                if isinstance(c_mod,(nn.Sequential,nn.ModuleList)):
                    try: c_mod=c_mod[idx]
                    except IndexError: return None
                else: return None
        return c_mod
    except Exception as e: print(f"Flask: Find layer err '{layer_n}': {e}"); return None

def compute_normgrad_order1_analysis(orig_m,in_t,tgt_ln,dev,eps=NORMGRAD_EPSILON_ANALYSIS,adv=False):
    m_grad=copy.deepcopy(orig_m).to(dev);m_grad.eval();m_grad.zero_grad()
    in_c_grad=in_t.clone().detach().to(dev).requires_grad_(True)
    try: out_o=m_grad(in_c_grad);scr_o=torch.sigmoid(out_o).mean();scr_o.backward()
    except Exception as e: print(f"Flask: Err NG1 grads {tgt_ln}: {e}"); del m_grad; return None
    p_grads={n:p_val.grad.data.clone().detach() for n,p_val in m_grad.named_parameters() if p_val.grad is not None}; del m_grad
    if not p_grads: print(f"Flask Warn: No NG1 grads {tgt_ln}.")
    m_prime=copy.deepcopy(orig_m).to(dev);m_prime.eval()
    with torch.no_grad():
        for n,p_val in m_prime.named_parameters():
            if n in p_grads: p_val.add_(p_grads[n]*eps) if adv else p_val.sub_(p_grads[n]*eps)
    tgt_mod=find_layer_analysis(m_prime,tgt_ln)
    if tgt_mod is None: print(f"Flask Err: No target '{tgt_ln}' prime."); del m_prime; return None
    acts_p,grads_p=None,None
    def fwd_h(m,i,o): nonlocal acts_p; acts_p=(o[0] if isinstance(o,(list,tuple)) else o).detach()
    def bwd_h(m,gi,go): nonlocal grads_p; grads_p=(go[0].detach() if go[0] is not None else None)
    h_f,h_b = tgt_mod.register_forward_hook(fwd_h),tgt_mod.register_full_backward_hook(bwd_h)
    in_c_prime=in_t.clone().detach().to(dev).requires_grad_(True); ng_map=None
    try:
        out_p=m_prime(in_c_prime);scr_p=torch.sigmoid(out_p).mean();m_prime.zero_grad();scr_p.backward()
        if acts_p is None or grads_p is None: print(f"Flask Err: No acts/grads NG1 {tgt_ln}.")
        else: ng_map=(torch.linalg.norm(acts_p,2,1,False)*torch.linalg.norm(grads_p,2,1,False)).squeeze(0).cpu().numpy()
    except Exception as e: print(f"Flask Err NG1 prime {tgt_ln}: {e}")
    finally:
        h_f.remove(); h_b.remove(); m_prime.zero_grad(); del m_prime
        if in_c_prime.grad is not None: in_c_prime.grad.zero_()
    return ng_map

def get_normgrad_order1_saliency_single_layer(m,img_b,d,tgt_ln,**kw):
    h,w=img_b.shape[2:]
    ng_map=compute_normgrad_order1_analysis(m, img_b[0].unsqueeze(0), tgt_ln, d,eps=kw.get('epsilon', NORMGRAD_EPSILON_ANALYSIS),adv=kw.get('adversarial', False))
    if ng_map is None: ng_map=np.zeros((h,w))
    elif ng_map.shape!=(h,w):
        try: ng_map=cv2.resize(ng_map,(w,h),interpolation=cv2.INTER_LINEAR)
        except Exception as e: print(f"Flask Warn: Resize NG1 {tgt_ln} err: {e}.");ng_map=np.zeros((h,w))
    return torch.from_numpy(ng_map).float().unsqueeze(0).to(d)

def normalize_heatmap_tensor(h_t):
    min_v,max_v=h_t.min(),h_t.max()
    return (h_t-min_v)/(max_v-min_v) if max_v-min_v>1e-8 else torch.zeros_like(h_t)

def get_combined_normgrad_order1_saliency(m,img_b,d,tgt_ll,status_update_fn=None,**kw):
    B,C,H,W=img_b.shape; valid_maps=[]
    if B!=1: print("Flask Warn: CombinedNG1 expects B=1.")
    total_l=len(tgt_ll)
    for i,ln in enumerate(tgt_ll):
        if status_update_fn: status_update_fn(f"NormGrad Layer: {ln[:30]}... ({i+1}/{total_l})")
        s_map=get_normgrad_order1_saliency_single_layer(m,img_b,d,ln,**kw)
        if s_map is not None and s_map.nelement()>0: valid_maps.append(normalize_heatmap_tensor(s_map.squeeze(0)))
    if valid_maps: return torch.mean(torch.stack(valid_maps,0),0).unsqueeze(0)
    print("Flask Warn: No valid NG1 maps."); return torch.zeros((1,H,W),device=d,dtype=torch.float32)

def get_binary_mask_analysis(m,img_t,d,thr=0.5,out_p=None):
    if out_p is not None: return ( (torch.from_numpy(out_p).to(d) if isinstance(out_p,np.ndarray) else out_p.to(d)) >thr).float().squeeze().cpu().numpy().astype(np.uint8)
    m.eval();
    with torch.no_grad():
        try: return (torch.sigmoid(m(img_t.to(d)))>thr).float().squeeze().cpu().numpy().astype(np.uint8)
        except Exception as e: print(f"Flask Err bin_mask: {e}"); traceback.print_exc(); return None

def extract_objects_from_mask(bin_m):
    if bin_m is None or not isinstance(bin_m,np.ndarray) or bin_m.ndim!=2: print("Flask: Invalid bin_mask."); return [],None,None
    nl,lbl,st,cent = cv2.connectedComponentsWithStats(bin_m,8)
    obj_l,total_px = [],bin_m.shape[0]*bin_m.shape[1]
    if nl<=1: return [],None,None
    for i in range(1,nl): obj_l.append({'object_id':i,'size_pixels':st[i,cv2.CC_STAT_AREA],'relative_size':(st[i,cv2.CC_STAT_AREA]/total_px) if total_px>0 else 0,'mean_index':(cent[i,1],cent[i,0]),'bbox':(st[i,cv2.CC_STAT_LEFT],st[i,cv2.CC_STAT_TOP],st[i,cv2.CC_STAT_WIDTH],st[i,cv2.CC_STAT_HEIGHT])})
    return obj_l,lbl,st

def calculate_object_saliency_means_analysis(obj_l,lbl_m,ng_h):
    if ng_h is None or lbl_m is None or (hasattr(lbl_m, 'shape') and hasattr(ng_h, 'shape') and lbl_m.shape!=ng_h.shape) :
        for o_item in obj_l: o_item['mean_saliency']=0.0 
        return obj_l
    for o_item in obj_l: 
        o_px=ng_h[lbl_m==o_item['object_id']]
        o_item['mean_saliency']=np.mean(o_px) if o_px.size>0 else 0.0
    return obj_l

def calculate_importance_scores_analysis(obj_l,h,w,cy_r=(35,172),cx_r=(44,200)):
    c_cy,c_cx = (cy_r[0]+cy_r[1])/2.0,(cx_r[0]+cx_r[1])/2.0; c_cent=np.array([c_cy,c_cx])
    corn=[np.array([0,0]),np.array([0,w]),np.array([h,0]),np.array([h,w])]; max_d=0
    if corn: max_d=max(np.linalg.norm(np.array([cy,cx])-c_cent) for cy,cx in corn)
    if max_d<1e-6: max_d=max(h,w,1.0)
    for o_item in obj_l: 
        oy,ox=o_item['mean_index']; o_cent=np.array([oy,ox]); dist=np.linalg.norm(o_cent-c_cent)
        in_c=(cy_r[0]<=oy<=cy_r[1]) and (cx_r[0]<=ox<=cx_r[1])
        o_item['importance_score']=max(1.0 if in_c else 1.0-min(dist/max_d,1.0),0.0)
    return obj_l

def size_penalty_sigmoid(rel_s,k=DEFAULT_SIGMOID_K,s_thr=DEFAULT_SIGMOID_THRESH):
    if rel_s<=0: return 0.0
    try: adj_in=k*(rel_s-s_thr); pen=1/(1+math.exp(-adj_in))
    except OverflowError: pen=0.0 if adj_in<0 else 1.0
    return min(max(pen,0.0),1.0)

def analyze_image_objects(m_an,img_t,d_use,ng_ll,status_update_fn=None,sig_k=DEFAULT_SIGMOID_K,sig_thr=DEFAULT_SIGMOID_THRESH,w_imp=0.3,w_sp=0.7,eps=NORMGRAD_EPSILON_ANALYSIS,adv=False):
    if status_update_fn: status_update_fn("Generating binary mask...")
    if img_t.shape[0]!=1: print("Flask Err: analyze_img_objs expects B=1."); return [],[],None,None,None,None
    h,w=img_t.shape[2:]; bin_m=get_binary_mask_analysis(m_an,img_t,d_use)
    if bin_m is None: print("Flask Err: Failed bin_mask."); return [],[],None,None,None,None
    if status_update_fn: status_update_fn("Extracting objects from mask...")
    objs_cca,lbl_m,stats = extract_objects_from_mask(bin_m)
    if not objs_cca:
        if status_update_fn: status_update_fn("No objects found after CCA.")
        return [],[],bin_m,lbl_m,stats,None
    if status_update_fn: status_update_fn("Starting NormGrad saliency map generation...")
    ng_h_map_t = get_combined_normgrad_order1_saliency(m_an,img_t,d_use,ng_ll,status_update_fn,epsilon=eps,adversarial=adv)
    if status_update_fn: status_update_fn("NormGrad saliency map generation complete.")
    ng_h_map_np = ng_h_map_t.squeeze(0).cpu().numpy()
    min_h_val,max_h_val = ng_h_map_np.min(),ng_h_map_np.max()
    ng_h_map_np = (ng_h_map_np-min_h_val)/(max_h_val-min_h_val) if max_h_val-min_h_val>1e-8 else np.zeros_like(ng_h_map_np)
    if status_update_fn: status_update_fn("Calculating object properties...")
    objs_proc = calculate_object_saliency_means_analysis(list(objs_cca),lbl_m,ng_h_map_np)
    objs_proc = calculate_importance_scores_analysis(objs_proc,h,w)
    fin_scores=[]
    for o_info in objs_proc:
        raw_s = o_info['relative_size']
        pen_s = size_penalty_sigmoid(o_info['relative_size'], k=sig_k, s_thr=sig_thr)
        ipo = (w_imp*o_info['importance_score'] + w_sp*pen_s); ipo=min(max(ipo,0.0),1.0)
        fin_scores.append({'object_id':o_info['object_id'],'mean_index':o_info['mean_index'],'RAW_SIZE':raw_s,'PENALTY_SIZE':pen_s,'IMPORTANCE':o_info['importance_score'],'CONFIDENT':o_info.get('mean_saliency',0.0),'individual_object_penalty':ipo,'bbox':o_info['bbox']})
    if status_update_fn: status_update_fn("Object property calculation complete.")
    return fin_scores,objs_cca,bin_m,lbl_m,stats,ng_h_map_np

def generate_analysis_text_summary(idx, obj_scores, q_all, tp_all, n_all, q_filt, tp_filt, n_filt, filt_ids, s_thr_val, cfg, bin_m_arg):
    def tex(s): return f"${s}$"
    def tex_display_html(latex_code):
        return f"<div class=\"tex2jax_process\">$${latex_code}$$</div>"

    title = f"Image Analysis Summary (Index: {idx})"    
    
    sigmoid_k_val = cfg.get('sigmoid_k', 0); sigmoid_thresh_val = cfg.get('sigmoid_thresh', 0)
    sigmoid_params_tex_inner = f"k = {sigmoid_k_val:.1f}, \\tau = {sigmoid_thresh_val:.4f}"
    weight_imp_val = cfg.get('weight_importance', 0); weight_sp_val = cfg.get('weight_size_penalty', 0)
    scoring_weights_tex_inner = f"w_L = {weight_imp_val:.2f}, w_S = {weight_sp_val:.2f}"
    params = [
        f"* Saliency Method: {cfg.get('normgrad_layer_display_name','N/A')}",
        f"* Sigmoid Parameters: {tex(sigmoid_params_tex_inner)}",
        f"* Scoring Weights: {tex(scoring_weights_tex_inner)}"
    ]
    
    formula_latex_str = r"S_i = w_{\text{Imp}} \cdot \text{Imp}_L + w_{\text{SizePen}} \cdot \text{Pen}_S"
    
    # Construct the formula section as a single multi-line string for Markdown
    # For the italicized line, let's use asterisks for italics, which are often more robust
    # or ensure it's clearly part of a paragraph.
    formula_section_md = f"""
**Scoring Methodology**:

{tex_display_html(formula_latex_str)}

*Saliency Confidence {tex('SalC')} is used for Score 2 filtering.* 
""" 
    # Changed from _..._ to *...* for italics. Alternatively, ensure no leading/trailing newlines
    # within the string that could break Markdown paragraph context for the _..._ version.
    # The {tex('SalC')} part for inline math should still work.

    sigma_s_i_tex = tex('\\Sigma S_i'); q_img_formula_tex = tex('Q_{img} = \\Pi (1 - S_i)')
    s1_lines = [f"### Score 1: All Objects ({n_all} detected)"]
    if obj_scores:
        for sd in obj_scores:
            raw_size_str = f'{sd["RAW_SIZE"]:.3f}'; pen_s_str = f'{sd["PENALTY_SIZE"]:.3f}'
            imp_l_str = f'{sd["IMPORTANCE"]:.3f}'; sal_c_str = f'{sd["CONFIDENT"]:.3f}'
            s_i_str = f'{sd["individual_object_penalty"]:.3f}'
            s1_lines.append(f"  * Obj {sd['object_id']}: {tex('RelS=' + raw_size_str)}, {tex('PenS=' + pen_s_str)}, {tex('ImpL=' + imp_l_str)}, {tex('SalC=' + sal_c_str)} $\\Rightarrow$ {tex('S_i=' + s_i_str)}")
    else: s1_lines.append("  * No objects detected.")
    q_all_val_str = f"{q_all:.3f}" if q_all is not None else "N/A"
    s1_lines.append(f"  * Total Penalty Sum ({sigma_s_i_tex} All): {tp_all:.3f}")
    s1_lines.append(f"  * **Overall Quality (All, {q_img_formula_tex}): {q_all_val_str}**")
    s2_lines = [f"### Score 2: High-Confidence Objects ({tex('SalC > ' + str(s_thr_val))} , {n_filt} objects)"]
    if n_filt > 0:
        s2_lines.append(f"  * Filtered Object IDs: {', '.join(map(str,sorted(filt_ids)))}")
        for sd in obj_scores:
            if sd['object_id'] in filt_ids:
                raw_size_str = f'{sd["RAW_SIZE"]:.3f}'; pen_s_str = f'{sd["PENALTY_SIZE"]:.3f}'
                imp_l_str = f'{sd["IMPORTANCE"]:.3f}'; sal_c_str = f'{sd["CONFIDENT"]:.3f}'
                s_i_str = f'{sd["individual_object_penalty"]:.3f}'
                s2_lines.append(f"  * Obj {sd['object_id']}: {tex('RelS=' + raw_size_str)}, {tex('PenS=' + pen_s_str)}, {tex('ImpL=' + imp_l_str)}, {tex('SalC=' + sal_c_str)} $\\Rightarrow$ {tex('S_i=' + s_i_str)}")
    elif obj_scores: s2_lines.append("  * No objects met the high-confidence saliency threshold.")
    else: s2_lines.append("  * No objects detected, so no high-confidence objects to filter.")
    q_filt_val_str = f"{q_filt:.3f}" if q_filt is not None else "N/A"
    s2_lines.append(f"  * Total Penalty Sum ({sigma_s_i_tex} Filtered): {tp_filt:.3f}")
    s2_lines.append(f"  * **Overall Quality (Filtered, {q_img_formula_tex}): {q_filt_val_str}**")
    if not obj_scores and bin_m_arg is None: params.append("\n**ANALYSIS FAILURE:** Mask generation error.")
    
    report_parts = [f"## {title}\n\n### Configuration Parameters"]
    report_parts.extend(params)
    report_parts.append("\n### Scoring Details") # Keep header separate
    report_parts.append(formula_section_md) # Add the whole block as one Markdown string
    report_parts.append("\n" + "\n".join(s1_lines))
    report_parts.append("\n" + "\n".join(s2_lines))
    
    return "\n".join(report_parts)

def format_analysis_for_llm(idx, cfg, obj_scores, n_all, ids_all, q_all, n_filt, filt_ids, q_filt, s_thr_val, bin_m_arg):
    def tex(s): return f"${s}$" 
    def tex_display(s): return f"$${s}$$"
    s_i_tex = tex('S_i'); q_img_tex = tex('Q_{img}'); w_L_tex = tex('w_L'); w_S_tex = tex('w_S')
    sigma_tex = tex('\\Sigma'); pi_tex = tex('\\Pi'); ge_0_6_tex = tex('\\ge 0.6')   
    individual_penalty_formula_tex = tex_display(r'S_i = w_{\text{Imp}} \cdot \text{Imp}_L + w_{\text{SizePen}} \cdot \text{Pen}_S')
    overall_quality_formula_tex = tex_display(r'Q_{img} = \Pi (1 - S_i)')
    prompt = [
        f"LLM Request: CXR Image Quality Assessment Report (Image Index: {idx})",
        "Your primary goal is to provide a clear, concise, and educational summary of an automated CXR image quality assessment.",
        "**Important: Start your response directly with the report content (e.g., with '## Overall Summary...'). Do not include any introductory phrases like 'Here's the report...' or 'Certainly, I can help...'.**",
        "The analysis identifies foreign objects and assesses their impact on image quality using factors like size, location, and model confidence (saliency).",
        "Please use Markdown for emphasis (like **bold text** or *italic text*). For lists, use standard Markdown bullet points (e.g., '- Item 1', '- Item 2') or numbered lists (e.g., '1. Item 1', '2. Item 2'), ensuring each list item starts on a new line.", 
        f"When referring to mathematical formulas or symbolic representations (like {s_i_tex} for individual object penalty, {q_img_tex} for overall image quality, {w_L_tex} for location weight, {w_S_tex} for size penalty weight, {sigma_tex} for sum, {pi_tex} for product), please use LaTeX notation (e.g., $S_i$, not S_i).",
        f"For example, the individual object penalty is calculated as: {individual_penalty_formula_tex}", 
        f"The overall image quality is calculated as: {overall_quality_formula_tex} for relevant objects.",
        "Saliency Score (referred to as 'SalC' or 'CONFIDENT' in data) indicates model confidence and is used for filtering objects for Score 2.",
        f"\n--- Configuration (Reference for your understanding) ---\n  - Importance Weight ({w_L_tex}): {cfg.get('weight_importance',0):.2f}\n  - Size Penalty Weight ({w_S_tex}): {cfg.get('weight_size_penalty',0):.2f}\n"
    ]
    if not obj_scores: prompt.extend(["\n--- Object Detection Summary ---", "No foreign objects were detected in this image."])
    else:
        prompt.append("\n--- Individual Detected Object Details ---")
        for sd in obj_scores:
            penalty_val_str = f"{sd['individual_object_penalty']:.3f}"; size_pen_val_str = f"{sd['PENALTY_SIZE']:.3f}"; imp_l_val_str = f"{sd['IMPORTANCE']:.3f}"; confident_val_str = f"{sd['CONFIDENT']:.3f}"
            prompt.append(f"  - Object ID {sd['object_id']}: Individual Penalty ({s_i_tex}) = {penalty_val_str}. Contributing factors: Size Penalty ({tex('Pen_S')}) = {size_pen_val_str}, Location Importance ({tex('Imp_L')}) = {imp_l_val_str}. Saliency Confidence (SalC) = {confident_val_str}.")
    def get_q_cat(s_val):
        if s_val is None: return "Not Assessed"; 
        if s_val > 0.8: return "Good"; 
        if s_val > 0.5: return "Acceptable/Borderline"; 
        return "Poor"
    q_all_val_str = f"{q_all:.3f}" if q_all is not None else "N/A"; cat_all = get_q_cat(q_all); q_img_all_tex = tex('Q_{img\\_all}')
    prompt.extend([f"\n\n--- IMAGE QUALITY ASSESSMENT REPORT (Image Index: {idx}) ---\n\n## Overall Summary of Image Quality\nThis report summarizes the automated CXR quality assessment. Two primary quality scores are provided: Score 1 (based on all detected objects) and Score 2 (based on high-confidence objects only).\n\n## Score 1: Quality Based on All Detected Objects ({n_all} objects)\n  - Calculated Overall Quality Score ({q_img_all_tex}): **{q_all_val_str}**\n  - Interpretation: Based on all potential foreign objects identified, the image quality is assessed as **'{cat_all}'**."])
    if n_all == 0 and bin_m_arg is not None: prompt.append("    - No foreign objects were detected. This typically results in a higher quality score due to fewer penalties.")
    elif n_all > 0: prompt.append(f"    - This score is derived from the combined impact of all {n_all} detected object(s). Each object's penalty ({s_i_tex}) contributes to a reduction in the overall quality score; higher penalties lead to lower quality.")
    q_filt_val_str = f"{q_filt:.3f}" if q_filt is not None else "N/A"; cat_filt = get_q_cat(q_filt); q_img_filt_tex = tex('Q_{img\\_filt}'); salc_filter_tex = tex(f'SalC > {s_thr_val:.2f}'); le_tex = tex("\\le")
    prompt.extend([f"\n## Score 2: Quality Based on High-Confidence Objects ({n_filt} objects)\n  - (Objects are filtered if their Saliency Confidence {tex('SalC')} {le_tex} {s_thr_val:.2f})\n  - Calculated Overall Quality Score ({q_img_filt_tex}): **{q_filt_val_str}**\n  - Interpretation: When considering only objects detected with high model confidence ({salc_filter_tex}), the image quality is assessed as **'{cat_filt}'**."])
    if n_filt==0 and n_all>0: prompt.append(f"    - No objects met the high-confidence threshold ({salc_filter_tex}). While some objects were detected (contributing to Score 1), none were deemed highly salient by the model.")
    elif n_filt==0 and n_all==0 and bin_m_arg is not None: prompt.append("    - No objects were detected overall, so the high-confidence score also reflects high quality (no penalties).")
    elif n_filt==0 and n_all==0 and bin_m_arg is None: prompt.append("    - Mask generation failed, so object presence could not be assessed for Score 2.")
    prompt.append("\n## Comparison and Detailed Insights")
    if q_all is not None and q_filt is not None:
        diff_thr,comp_det=0.1,""; q_all_comp_str = f"{q_all:.3f}"; q_filt_comp_str = f"{q_filt:.3f}"
        if abs(q_all-q_filt)>diff_thr:
            comp="significantly different"
            comp_det = f"Score 2 ({q_img_filt_tex} = {q_filt_comp_str}) is greater than Score 1 ({q_img_all_tex} = {q_all_comp_str}). This suggests that the objects lowering the overall quality in Score 1 were mostly of low confidence. The assessment based on high-confidence findings indicates better quality." if q_filt>q_all else f"Score 2 ({q_img_filt_tex} = {q_filt_comp_str}) is less than or similar to Score 1 ({q_img_all_tex} = {q_all_comp_str}). This implies that high-confidence objects significantly impact the quality, or that low-confidence objects did not heavily penalize Score 1 to begin with."
        else: comp="similar"; comp_det="The two quality scores are similar. This indicates that the high-confidence objects largely represent the overall detected landscape in terms of impact, or that any low-confidence objects had minimal impact on Score 1."
        prompt.extend([f"The two quality scores ({q_all_comp_str} vs. {q_filt_comp_str}) are {comp}.",f"  - {comp_det}"])
    if obj_scores:
        prompt.append(f"\n### Explanation of Individual Object Penalties ({s_i_tex}):")
        for sd in obj_scores:
            s_i_val = sd['individual_object_penalty']; obj_id = sd['object_id']; factors = []
            imp_val = sd['IMPORTANCE']; pen_s_val = sd['PENALTY_SIZE']; raw_s_val = sd['RAW_SIZE']
            factors.append(f"critical location ({tex('Imp_L')}={imp_val:.2f})" if imp_val >= 0.8 else (f"less critical location ({tex('Imp_L')}={imp_val:.2f})" if imp_val <= 0.3 else f"moderate location impact ({tex('Imp_L')}={imp_val:.2f})"))
            factors.append(f"large relative size ({tex('Pen_S')}={pen_s_val:.2f})" if pen_s_val >= 0.5 else (f"small relative size ({tex('Pen_S')}={pen_s_val:.2f})" if pen_s_val <= 0.1 and raw_s_val > 0 else f"moderate relative size ({tex('Pen_S')}={pen_s_val:.2f})"))
            expl_s_i = ""
            if s_i_val >= 0.7: expl_s_i = f"Object {obj_id} has a very high penalty ({s_i_tex}={s_i_val:.3f}). This is primarily due to its " + (" and ".join(factors) + "." if factors else "combination of size and location.")
            elif s_i_val >= 0.4: expl_s_i = f"Object {obj_id} has a notable penalty ({s_i_tex}={s_i_val:.3f}), influenced by its " + (" and ".join(factors) + "." if factors else "combination of size and location.")
            elif s_i_val > 0: expl_s_i = f"Object {obj_id} has a relatively low penalty ({s_i_tex}={s_i_val:.3f}), indicating it is less impactful due to its " + (" or ".join(factors) + "." if factors else "combination of size and location.")
            else: expl_s_i = f"Object {obj_id} has no penalty ({s_i_tex}={s_i_val:.3f}), likely due to being very small or in an optimal location."
            if expl_s_i: prompt.append(f"  - {expl_s_i} Its Saliency Confidence (SalC) is {sd['CONFIDENT']:.3f}.")
    prompt.append("\n## Recommendations (Based on Automated Analysis)"); man_rev = False
    if q_all is not None:
        if q_all <= 0.5: prompt.append(f"  - **Manual review is strongly suggested** due to a poor Score 1 ({q_img_all_tex} = {q_all_val_str}). Focus on investigating objects with high penalty scores ({s_i_tex})."); man_rev = True
        elif q_all <= 0.8: prompt.append(f"  - Manual review may be warranted due to a borderline Score 1 ({q_img_all_tex} = {q_all_val_str}). Check objects with significant penalties."); man_rev = True
    if obj_scores:
        high_pen_objs = [o for o in obj_scores if o['individual_object_penalty'] >= 0.6]; high_sal_high_pen_objs = [o for o in high_pen_objs if o['object_id'] in filt_ids]
        if high_sal_high_pen_objs: obj_ids_str = ", ".join([str(o['object_id']) for o in high_sal_high_pen_objs]); prompt.append(f"  - Object(s) **{obj_ids_str}** are particularly noteworthy as they have both high penalty scores ({s_i_tex} {ge_0_6_tex}) and high Saliency Confidence. Prioritize these for review.")
        elif high_pen_objs and not man_rev: obj_ids_str = ", ".join([str(o['object_id']) for o in high_pen_objs]); prompt.append(f"  - Consider reviewing Object(s) **{obj_ids_str}**, which have high penalty scores ({s_i_tex} {ge_0_6_tex}), even if their Saliency Confidence was lower; they might still be relevant.")
    if not man_rev and (not obj_scores or (q_all is not None and q_all > 0.8)): prompt.append("  - No specific objects are flagged as highly problematic from this automated view. The image quality appears generally good based on the analysis metrics.")
    prompt.extend(["\n**Disclaimer:** This is an automated analysis providing insights based on configured parameters. Clinical correlation and expert radiological review are essential for any diagnostic conclusions."])
    return "\n".join(prompt)

def plot_to_numpy_array(fig):
    if fig is None: return np.zeros((IMG_SIZE_ANALYSIS, IMG_SIZE_ANALYSIS, 3), dtype=np.uint8)
    io_buf = io.BytesIO(); array_np = np.zeros((IMG_SIZE_ANALYSIS, IMG_SIZE_ANALYSIS, 3), dtype=np.uint8)
    try:
        fig.canvas.draw(); fig.savefig(io_buf, format='png', bbox_inches='tight', pad_inches=0); io_buf.seek(0)
        if io_buf.getbuffer().nbytes > 0:
            img_pil = Image.open(io_buf); img_rgb_pil = img_pil.convert('RGB'); array_np_temp = np.array(img_rgb_pil, dtype=np.uint8)
            if array_np_temp.size > 0 and array_np_temp.shape[0] > 10 and array_np_temp.shape[1] > 10: array_np = array_np_temp
            else: print("Flask WARN plot_to_numpy_array: Generated array too small.")
        else: print("Flask WARN plot_to_numpy_array: io_buf empty after savefig.")
    except Exception as e: print(f"Flask ERROR in plot_to_numpy_array: {e}"); traceback.print_exc()
    finally: io_buf.close(); plt.close(fig)
    return array_np

def create_mpl_object_segmentation_image_array(orig_img, lbl_mask, obj_scores, bin_fallback):
    if orig_img is None: return np.zeros((IMG_SIZE_ANALYSIS, IMG_SIZE_ANALYSIS, 3), dtype=np.uint8)
    is_gray = orig_img.ndim == 3 and orig_img.shape[2] == 1
    disp_img = orig_img.squeeze(axis=2) if is_gray else orig_img; cmap_disp = 'gray' if is_gray else None
    fig, ax = plt.subplots(figsize=(IMG_SIZE_ANALYSIS / 100., IMG_SIZE_ANALYSIS / 100.), dpi=100)
    ax.imshow(disp_img, cmap=cmap_disp); ax.axis('off')
    if obj_scores:
        n_obj = len(obj_scores); cmap_tab20 = plt.get_cmap('tab20', n_obj if n_obj > 0 else 1)
        overlay = np.zeros((*lbl_mask.shape, 4), dtype=float) if lbl_mask is not None else None
        for i, sd in enumerate(obj_scores):
            obj_id = sd['object_id']; rgba_color = cmap_tab20(i % cmap_tab20.N)
            if overlay is not None and lbl_mask is not None and np.any(lbl_mask == obj_id): overlay[lbl_mask == obj_id] = rgba_color
            x, y, w, h = sd['bbox']
            ax.add_patch(patches.Rectangle((x, y), w, h, lw=1.5, ec=rgba_color[:3], fc='none'))
            ax.text(x + 3, y + 10, f"Obj {obj_id}", color='white', weight='bold', fontsize=6, bbox=dict(boxstyle="round,pad=0.2", alpha=0.75, facecolor=rgba_color[:3], edgecolor='black', linewidth=0.4))
        if overlay is not None: ax.imshow(overlay, alpha=0.45)
    else:
        text_props = {'ha': 'center', 'va': 'center', 'fontsize': 8, 'bbox': {'fc': 'w', 'alpha': 0.7, 'pad': 0.2}}
        if bin_fallback is not None:
            if np.any(bin_fallback): ax.imshow(bin_fallback, cmap='Greys', alpha=0.5); ax.text(IMG_SIZE_ANALYSIS / 2, IMG_SIZE_ANALYSIS / 2, "Binary Mask (No Objects Scored)", color='darkorange', **text_props)
            else: ax.text(IMG_SIZE_ANALYSIS / 2, IMG_SIZE_ANALYSIS / 2, "No Objects Detected (Empty Mask)", color='red', **text_props)
        else: ax.text(IMG_SIZE_ANALYSIS / 2, IMG_SIZE_ANALYSIS / 2, "Object Masking Failed", color='red', **text_props)
    fig.tight_layout(pad=0); return plot_to_numpy_array(fig)

def create_mpl_saliency_overlay_image_array(orig_img, heat_map, obj_scores, s_thr_val):
    if orig_img is None: return np.zeros((IMG_SIZE_ANALYSIS, IMG_SIZE_ANALYSIS, 3), dtype=np.uint8)
    is_gray = orig_img.ndim == 3 and orig_img.shape[2] == 1
    disp_img = orig_img.squeeze(axis=2) if is_gray else orig_img; cmap_disp = 'gray' if is_gray else None
    fig, ax = plt.subplots(figsize=(IMG_SIZE_ANALYSIS / 100., IMG_SIZE_ANALYSIS / 100.), dpi=100)
    ax.imshow(disp_img, cmap=cmap_disp)
    if heat_map is not None: ax.imshow(heat_map, cmap='jet', alpha=0.6, vmin=0, vmax=1)
    else: ax.text(IMG_SIZE_ANALYSIS / 2, IMG_SIZE_ANALYSIS * 0.1, "Saliency Map N/A", ha='center', va='center', fontsize=7, color='gray', bbox=dict(boxstyle="round,pad=0.1", fc="white", ec="gray", alpha=0.5, lw=0.5))
    ax.axis('off')
    if obj_scores:
        for sd in obj_scores:
            obj_id, bbox = sd['object_id'], sd['bbox']; x, y, w, h = bbox
            is_above = sd['CONFIDENT'] > s_thr_val; box_c = 'lime' if is_above else 'orangered'
            ax.add_patch(patches.Rectangle((x, y), w, h, lw=1.8, ec=box_c, fc='none', alpha=0.9))
            ax.text(x + 3, y + 10, f"Obj {obj_id}", color='white', weight='bold', fontsize=6, bbox=dict(boxstyle="round,pad=0.2", facecolor=box_c, alpha=0.7, edgecolor='black', linewidth=0.4))
    else: ax.text(IMG_SIZE_ANALYSIS / 2, IMG_SIZE_ANALYSIS / 2, "No Objects Detected", ha='center', va='center', fontsize=10, color='red', weight='bold', bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="red", alpha=0.8, lw=1))
    fig.tight_layout(pad=0); return plot_to_numpy_array(fig)

def perform_full_image_analysis( analysis_model_param, analysis_test_dataset_param, analysis_device_param, image_index_to_analyze_param: int, normgrad_target_layers_list_param, analysis_config_dict_param: dict, status_update_fn=None):
    def _upd_status_local(description_str):
        if status_update_fn: status_update_fn(description_str)
    _upd_status_local(f"Loading image index: {image_index_to_analyze_param}...")
    sigmoid_k = analysis_config_dict_param.get('SIGMOID_K_PARAM',DEFAULT_SIGMOID_K); sigmoid_thresh_val = analysis_config_dict_param.get('SIGMOID_THRESH_PARAM',DEFAULT_SIGMOID_THRESH)
    weight_importance_val = analysis_config_dict_param.get('WEIGHT_IMPORTANCE',0.3); weight_size_penalty_val = analysis_config_dict_param.get('WEIGHT_SIZE_PENALTY',0.7)
    saliency_thresh_val = analysis_config_dict_param.get('SALIENCY_FILTER_THRESHOLD',0.3); ng_eps_val = analysis_config_dict_param.get('NORMGRAD_EPSILON',NORMGRAD_EPSILON_ANALYSIS)
    ng_adv_val = analysis_config_dict_param.get('NORMGRAD_ADVERSARIAL',False); ds_len = len(analysis_test_dataset_param); curr_idx = int(image_index_to_analyze_param)
    if not (0<=curr_idx<ds_len): err_msg = f"Index {curr_idx} out of bounds for dataset size {ds_len}."; print(f"Flask: {err_msg}"); _upd_status_local(f"ERROR: {err_msg}"); raise IndexError(err_msg)    
    img_t_orig, _ = analysis_test_dataset_param[curr_idx]; _upd_status_local(f"Image tensor loaded (shape: {img_t_orig.shape}).")
    if img_t_orig.ndim==3: img_b=img_t_orig.unsqueeze(0).to(analysis_device_param)
    elif img_t_orig.ndim==4 and img_t_orig.shape[0]==1: img_b=img_t_orig.to(analysis_device_param)
    else: err_msg = f"Img tensor shape {img_t_orig.shape} unexpected."; print(f"Flask: {err_msg}"); _upd_status_local(f"ERROR: {err_msg}"); return None    
    if isinstance(img_t_orig, torch.Tensor): img_np_temp = img_t_orig.cpu().permute(1,2,0).numpy()
    else: 
        img_np_temp = img_t_orig
        if img_np_temp.ndim == 3 and img_np_temp.shape[0] == 3: img_np_temp = img_np_temp.transpose(1,2,0)
    orig_img_np_float_unclipped = img_np_temp; orig_img_np_float_clipped_for_uint8 = np.clip(img_np_temp,0,1)
    if orig_img_np_float_clipped_for_uint8.ndim == 2: orig_img_np_rgb_for_uint8 = np.stack((orig_img_np_float_clipped_for_uint8,)*3, axis=-1)
    elif orig_img_np_float_clipped_for_uint8.ndim == 3 and orig_img_np_float_clipped_for_uint8.shape[-1] == 1: orig_img_np_rgb_for_uint8 = np.concatenate((orig_img_np_float_clipped_for_uint8,)*3, axis=-1)
    elif orig_img_np_float_clipped_for_uint8.ndim == 3 and orig_img_np_float_clipped_for_uint8.shape[-1] == 3: orig_img_np_rgb_for_uint8 = orig_img_np_float_clipped_for_uint8
    else: orig_img_np_rgb_for_uint8 = np.zeros((IMG_SIZE_ANALYSIS, IMG_SIZE_ANALYSIS, 3), dtype=np.float32)    
    original_image_np_display_uint8 = (orig_img_np_rgb_for_uint8 * 255).astype(np.uint8); _upd_status_local("Original image processed.")
    if orig_img_np_float_unclipped.ndim == 2: mpl_base_image_float = np.stack((orig_img_np_float_unclipped,)*3, axis=-1)
    elif orig_img_np_float_unclipped.ndim == 3 and orig_img_np_float_unclipped.shape[-1] == 1: mpl_base_image_float = np.concatenate((orig_img_np_float_unclipped,)*3, axis=-1)
    elif orig_img_np_float_unclipped.ndim == 3 and orig_img_np_float_unclipped.shape[-1] == 3: mpl_base_image_float = orig_img_np_float_unclipped
    else: mpl_base_image_float = np.zeros((IMG_SIZE_ANALYSIS, IMG_SIZE_ANALYSIS, 3), dtype=np.float32)    
    obj_s_all, _, bin_m, lbl_m, _, ng_hmap_norm = analyze_image_objects( analysis_model_param,img_b,analysis_device_param,normgrad_target_layers_list_param, status_update_fn=_upd_status_local, sig_k=sigmoid_k,sig_thr=sigmoid_thresh_val, w_imp=weight_importance_val,w_sp=weight_size_penalty_val, eps=ng_eps_val,adv=ng_adv_val); _upd_status_local("Object analysis complete.")    
    _upd_status_local("Calculating quality scores..."); q_all,p_sum_all,ids_all=1.0,[],[]
    if obj_s_all:
        for sd in obj_s_all: ipo=sd['individual_object_penalty']; p_sum_all.append(ipo);ids_all.append(sd['object_id']);q_all*=(1.0-ipo)
        q_all=min(max(q_all,0.0),1.0)
    tot_p_all,n_all=sum(p_sum_all),len(ids_all)    
    if not obj_s_all and bin_m is None: q_all=0.0
    elif not obj_s_all and bin_m is not None: q_all=1.0    
    q_filt,p_sum_filt,ids_filt=1.0,[],[]; sfd = [o for o in obj_s_all if o['CONFIDENT']>saliency_thresh_val] if obj_s_all else []
    if sfd:
        for sd_f in sfd: ipo_f=sd_f['individual_object_penalty']; p_sum_filt.append(ipo_f);ids_filt.append(sd_f['object_id']);q_filt*=(1.0-ipo_f)
        q_filt=min(max(q_filt,0.0),1.0)
    tot_p_filt,n_filt=sum(p_sum_filt),len(ids_filt)    
    if not obj_s_all and bin_m is None: q_filt=0.0
    elif not sfd and bin_m is not None: q_filt=1.0
    _upd_status_local("Quality scores calculated.")    
    _upd_status_local("Generating visualization: Segmentation overlay..."); mpl_seg=create_mpl_object_segmentation_image_array(mpl_base_image_float,lbl_m,obj_s_all,bin_m)
    _upd_status_local("Generating visualization: Saliency overlay..."); mpl_sal=create_mpl_saliency_overlay_image_array(mpl_base_image_float,ng_hmap_norm,obj_s_all,saliency_thresh_val)
    _upd_status_local("Visualizations generated.")    
    _upd_status_local("Generating text summary report...")
    plot_cfg={'normgrad_layer_display_name':f"Comb NG O1 ({len(normgrad_target_layers_list_param)} L)", "sigmoid_k":sigmoid_k, "sigmoid_thresh":sigmoid_thresh_val, "weight_importance":weight_importance_val, "weight_size_penalty":weight_size_penalty_val}
    txt_sum=generate_analysis_text_summary(curr_idx,obj_s_all,q_all,tot_p_all,n_all,q_filt,tot_p_filt,n_filt,ids_filt,saliency_thresh_val,plot_cfg,bin_m); _upd_status_local("Text summary report generated.")    
    llm_prompt_data = None
    if analysis_config_dict_param.get('USE_LLM_EXPLANATION',False):
        _upd_status_local("Preparing data for LLM.")
        llm_prompt_data = {"idx": curr_idx, "cfg": plot_cfg, "obj_scores": obj_s_all, "n_all": n_all, "ids_all": ids_all, "q_all": q_all, "n_filt": n_filt, "filt_ids": ids_filt, "q_filt": q_filt, "s_thr_val": saliency_thresh_val, "bin_m_arg": bin_m}    
    results_dict = { "image_index_analyzed":curr_idx, "original_image_np_display":original_image_np_display_uint8, "mpl_object_segmentation_overlay":mpl_seg, "mpl_saliency_overlay":mpl_sal, "quality_metrics":{"overall_image_quality_all":q_all, "num_objects_all":n_all, "overall_image_quality_filtered":q_filt, "num_objects_filtered":n_filt}, "text_summary_report":txt_sum, "llm_prompt_data": llm_prompt_data, "llm_explanation_text": "LLM processing handled by server route." if llm_prompt_data else "LLM Disabled/Not Applicable", "llm_status": "Pending in Route" if llm_prompt_data else "LLM Disabled", "llm_error":None }; _upd_status_local("Core analysis processing complete.")
    return results_dict

g_device_flask = None; g_model_flask = None; g_test_dataset_flask = None
g_max_image_index_flask = -1; g_llm_instance_flask = None

def load_resources():
    global g_device_flask, g_model_flask, g_test_dataset_flask, g_max_image_index_flask, g_llm_instance_flask
    if g_model_flask is not None and g_test_dataset_flask is not None: print("Flask: Resources appear to be already loaded."); return
    print("Flask: Initializing and loading resources..."); g_device_flask = torch.device("cuda" if torch.cuda.is_available() else "cpu"); print(f"Flask: Using device: {g_device_flask}")
    try:
        csv_path = os.path.join(DATA_BASE_PATH, TEST_CSV_NAME); img_dir_name = os.path.splitext(TEST_CSV_NAME)[0]; img_dir = os.path.join(DATA_BASE_PATH, img_dir_name); print(f"Flask: Attempting to load dataset. CSV: {csv_path}, Image Dir: {img_dir}")
        if not os.path.exists(csv_path): print(f"Flask ERROR: CSV file not found at {csv_path}")
        if not os.path.isdir(img_dir):
            img_dir_alt1 = os.path.join(DATA_BASE_PATH, "images"); img_dir_alt2 = DATA_BASE_PATH
            if os.path.isdir(img_dir_alt1): img_dir = img_dir_alt1; print(f"Flask: Using alt image directory: {img_dir}")
            elif os.path.isdir(img_dir_alt2) and any(f.lower().endswith(('.png', '.jpg', '.jpeg')) for f in os.listdir(img_dir_alt2) if os.path.isfile(os.path.join(DATA_BASE_PATH,f))): img_dir = DATA_BASE_PATH; print(f"Flask: Using DATA_BASE_PATH as image directory: {img_dir}")
            else: print(f"Flask ERROR: Image directory not found at {img_dir} or alternatives.")
        g_test_dataset_flask = ObjectCXRSataset(csv_path, img_dir, transform=test_transform_global, img_size=(IMG_SIZE, IMG_SIZE))
        if g_test_dataset_flask and len(g_test_dataset_flask) > 0: g_max_image_index_flask = len(g_test_dataset_flask) - 1; print(f"Flask: Dataset loaded successfully with {len(g_test_dataset_flask)} images. Max index: {g_max_image_index_flask}")
        else: print("Flask ERROR: Dataset is empty or failed to load after initialization."); g_max_image_index_flask = -1; g_test_dataset_flask = None
        if os.path.exists(TRAINED_MODEL_PATH):
            g_model_flask = SegmentationModel().to(g_device_flask); g_model_flask.load_state_dict(torch.load(TRAINED_MODEL_PATH, map_location=g_device_flask)); g_model_flask.eval(); print(f"Flask: Segmentation model loaded from {TRAINED_MODEL_PATH}")
        else: print(f"Flask ERROR: Trained model path not found: {TRAINED_MODEL_PATH}."); g_model_flask = None
        if analysis_config_global.get("USE_LLM_EXPLANATION", False) and GPT4ALL_AVAILABLE:
            print(f"Flask: Attempting to load LLM: {analysis_config_global['LLM_MODEL_NAME']} on {analysis_config_global['LLM_DEVICE']}")
            try: g_llm_instance_flask = GPT4All(model_name=analysis_config_global['LLM_MODEL_NAME'], device=analysis_config_global.get('LLM_DEVICE', 'cpu'), allow_download=True); print("Flask: LLM instance loaded successfully.")
            except Exception as e_llm: print(f"Flask ERROR loading LLM: {e_llm}"); g_llm_instance_flask = None
        elif not GPT4ALL_AVAILABLE and analysis_config_global.get("USE_LLM_EXPLANATION"): print("Flask: LLM explanations enabled in config, but gpt4all library is not available."); analysis_config_global["USE_LLM_EXPLANATION"] = False
    except Exception as e: print(f"Flask: General error during resource loading: {e}"); traceback.print_exc(); g_model_flask = None; g_test_dataset_flask = None; g_max_image_index_flask = -1; g_llm_instance_flask = None
    print("Flask: Resource loading process finished.")

app = Flask(__name__)
NGROK_AUTH_TOKEN = "2wob6CRL9i6tMgymnUqlg52VlbL_5RyxJFxq6RBzTtwHRjsFW" 
public_url_global = None 

def start_ngrok():
    global public_url_global
    if NGROK_AUTH_TOKEN == "YOUR_NGROK_AUTH_TOKEN" or not NGROK_AUTH_TOKEN: print("Flask WARNING: NGROK_AUTH_TOKEN is not set."); return None
    try:
        conf.get_default().auth_token = NGROK_AUTH_TOKEN
        for tunnel in ngrok.get_tunnels():
            if tunnel.conf.get("addr") and "5001" in tunnel.conf.get("addr"): print(f"Flask: Disconnecting existing ngrok tunnel: {tunnel.public_url}"); ngrok.disconnect(tunnel.public_url)
        tunnel = ngrok.connect(5001, bind_tls=True)
        public_url_global = tunnel.public_url
        print(f"Flask: * ngrok tunnel \"{public_url_global}\" -> \"http://127.0.0.1:5001\""); return public_url_global
    except Exception as e: print(f"Flask ERROR: pyngrok failed: {e}"); return None

@app.route('/')
def index_route(): return render_template('index.html', public_url=public_url_global )

@app.route('/get_initial_config', methods=['GET'])
def get_initial_config():
    return jsonify({
        "max_index": g_max_image_index_flask,
        "dataset_message": "Dataset loaded." if g_max_image_index_flask >=0 and g_test_dataset_flask is not None else "Dataset not loaded or empty on server.",
        "dataset_ok": g_max_image_index_flask >=0 and g_test_dataset_flask is not None,
        "llm_available": g_llm_instance_flask is not None,
        "llm_initially_enabled": analysis_config_global.get("USE_LLM_EXPLANATION", False) and (g_llm_instance_flask is not None),
        "public_url": public_url_global
    })

@app.route('/analyze', methods=['POST'])
def analyze_image_route():
    if g_model_flask is None or g_test_dataset_flask is None: return jsonify({"error": "Model or dataset not loaded on server."}), 503
    data = request.get_json(); image_index = data.get('image_index'); use_llm_from_client = data.get('use_llm', False)
    if image_index is None: return jsonify({"error": "No image_index provided."}), 400
    try:
        image_index = int(image_index)
        if not (0 <= image_index <= g_max_image_index_flask):
            return jsonify({"error": f"Image index {image_index} out of bounds (0-{g_max_image_index_flask})."}), 400
        
        print(f"Flask: Analyzing image index: {image_index}")
        def server_status_update(msg): 
            print(f"ANALYSIS_LOG (Img {image_index}): {msg}")

        current_analysis_config = analysis_config_global.copy(); current_analysis_config["USE_LLM_EXPLANATION"] = use_llm_from_client and (g_llm_instance_flask is not None)
        analysis_results = perform_full_image_analysis(
            analysis_model_param=g_model_flask, analysis_test_dataset_param=g_test_dataset_flask, analysis_device_param=g_device_flask, 
            image_index_to_analyze_param=image_index, normgrad_target_layers_list_param=NORMGRAD_TARGET_LAYERS_LIST,
            analysis_config_dict_param=current_analysis_config, status_update_fn=server_status_update
        )
        llm_explanation_text = "LLM explanation disabled or not requested."; llm_final_status = "LLM Disabled"
        if current_analysis_config["USE_LLM_EXPLANATION"] and g_llm_instance_flask and analysis_results.get("llm_prompt_data"):
            print(f"Flask: Generating LLM explanation for image {image_index}...")
            try:
                llm_data = analysis_results["llm_prompt_data"]
                llm_prompt = format_analysis_for_llm( llm_data["idx"], llm_data["cfg"], llm_data["obj_scores"], llm_data["n_all"], llm_data["ids_all"], llm_data["q_all"], llm_data["n_filt"], llm_data["filt_ids"], llm_data["q_filt"], llm_data["s_thr_val"], llm_data["bin_m_arg"] )
                with g_llm_instance_flask.chat_session(): llm_explanation_text = g_llm_instance_flask.generate(prompt=llm_prompt, max_tokens=1500, temp=0.7)
                llm_final_status = "LLM Success"; print(f"Flask: LLM explanation generated for image {image_index}.")
            except Exception as e_llm: print(f"Flask ERROR during LLM generation: {e_llm}"); traceback.print_exc(); llm_explanation_text = f"Error during LLM generation: {str(e_llm)}"; llm_final_status = f"LLM Error"
        elif current_analysis_config["USE_LLM_EXPLANATION"] and not g_llm_instance_flask: llm_explanation_text = "LLM was requested but is not available/loaded on the server."; llm_final_status = "LLM Not Available"
        def np_to_base64(img_array):
            if img_array is None: return None
            try:
                if img_array.dtype != np.uint8:
                    if img_array.max() <= 1.0 and img_array.min() >=0.0 : img_array = (img_array * 255)
                    img_array = np.clip(img_array, 0, 255).astype(np.uint8)
                img_pil = Image.fromarray(img_array); buffered = io.BytesIO(); img_pil.save(buffered, format="PNG"); return base64.b64encode(buffered.getvalue()).decode('utf-8')
            except Exception as e_img: print(f"Flask: Error converting image to base64: {e_img}"); return None
        response_data = {
            "original_image": np_to_base64(analysis_results.get("original_image_np_display")), "segmentation_overlay": np_to_base64(analysis_results.get("mpl_object_segmentation_overlay")),
            "saliency_overlay": np_to_base64(analysis_results.get("mpl_saliency_overlay")), "quality_metrics": analysis_results.get("quality_metrics"),
            "text_summary_report": analysis_results.get("text_summary_report"), "llm_explanation_text": llm_explanation_text, "llm_status": llm_final_status,
            "image_index_analyzed": analysis_results.get("image_index_analyzed")
        }
        return jsonify(response_data)
    except IndexError as ie: print(f"Flask Error (IndexError): {str(ie)}"); return jsonify({"error": str(ie)}), 400
    except Exception as e: print(f"Flask: General error during analysis for index {image_index}: {e}"); traceback.print_exc(); return jsonify({"error": "An internal server error occurred during analysis."}), 500

if __name__ == '__main__':
    load_resources()
    if os.environ.get("WERKZEUG_RUN_MAIN") != "true":
        start_ngrok() 
    else: # In reloader process
        # Attempt to get existing tunnel info if pyngrok manages tunnels globally per process tree
        # This might not always work perfectly with pyngrok's internal state after reloads
        all_tunnels = ngrok.get_tunnels()
        if all_tunnels:
            public_url_global = all_tunnels[0].public_url
            print(f"Flask (Reloader): Re-fetched ngrok tunnel: {public_url_global}")
        
    if g_model_flask is None or g_test_dataset_flask is None: print("\nFlask CRITICAL ERROR: Model or Dataset failed to load.")
    print(f"\nFlask server starting... Access locally at http://localhost:5001 or http://0.0.0.0:5001")
    if public_url_global: print(f"Publicly accessible via ngrok at: {public_url_global}")
    else: print("ngrok tunnel not started or NGROK_AUTH_TOKEN not set.")
    app.run(debug=True, host='0.0.0.0', port=5001, use_reloader=False) # use_reloader=False recommended with pyngrok