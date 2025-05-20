# app/services/image_processor_service.py
import copy
import cv2
import torch
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import io
from PIL import Image
import math

from config import (
    NORMGRAD_EPSILON_ANALYSIS, DEFAULT_SIGMOID_K, DEFAULT_SIGMOID_THRESH,
    IMG_SIZE_ANALYSIS
)
# Assuming model_handler_service provides the get_segmentation_model method
# No direct import of ModelHandlerService itself, but its instance will be passed

class ImageProcessorService:
    def __init__(self, model_handler_instance, analysis_img_size=IMG_SIZE_ANALYSIS, device_str=None):
        self.model_handler = model_handler_instance # Expects an initialized ModelHandlerService instance
        self.analysis_img_size = analysis_img_size 
        self.device = torch.device(device_str if device_str else ("cuda" if torch.cuda.is_available() else "cpu"))

    def _find_layer_analysis(self, model_s, layer_n): 
        s_base, c_path = model_s, layer_n
        if hasattr(model_s, 'model') and isinstance(model_s.model, nn.Module) and not layer_n.startswith('model.'): s_base = model_s.model
        elif layer_n.startswith('model.'): c_path = layer_n[len('model.'):]
        c_mod = s_base
        try:
            for p_item in c_path.split('.'):
                if hasattr(c_mod, p_item): c_mod = getattr(c_mod, p_item)
                else:
                    try: idx = int(p_item)
                    except ValueError: return None
                    if isinstance(c_mod, (nn.Sequential, nn.ModuleList)):
                        try: c_mod = c_mod[idx]
                        except IndexError: return None
                    else: return None
            return c_mod
        except Exception as e: print(f"ImageProcessor Service: Find layer err '{layer_n}': {e}"); return None

    def _compute_normgrad_order1_analysis(self, orig_m, in_t, tgt_ln, dev, eps=NORMGRAD_EPSILON_ANALYSIS, adv=False): 
        m_grad = copy.deepcopy(orig_m).to(dev); m_grad.eval(); m_grad.zero_grad()
        in_c_grad = in_t.clone().detach().to(dev).requires_grad_(True)
        try: out_o = m_grad(in_c_grad); scr_o = torch.sigmoid(out_o).mean(); scr_o.backward()
        except Exception as e: print(f"ImageProcessor Service: Err NG1 grads {tgt_ln}: {e}"); del m_grad; return None
        p_grads = {n: p_val.grad.data.clone().detach() for n, p_val in m_grad.named_parameters() if p_val.grad is not None}; del m_grad
        if not p_grads: print(f"ImageProcessor Service Warn: No NG1 grads {tgt_ln}.")
        m_prime = copy.deepcopy(orig_m).to(dev); m_prime.eval()
        with torch.no_grad():
            for n, p_val in m_prime.named_parameters():
                if n in p_grads: p_val.add_(p_grads[n] * eps) if adv else p_val.sub_(p_grads[n] * eps)
        tgt_mod = self._find_layer_analysis(m_prime, tgt_ln)
        if tgt_mod is None: print(f"ImageProcessor Service Err: No target '{tgt_ln}' prime."); del m_prime; return None
        acts_p, grads_p = None, None
        def fwd_h(m, i, o): nonlocal acts_p; acts_p = (o[0] if isinstance(o, (list, tuple)) else o).detach()
        def bwd_h(m, gi, go): nonlocal grads_p; grads_p = (go[0].detach() if go[0] is not None else None)
        h_f, h_b = tgt_mod.register_forward_hook(fwd_h), tgt_mod.register_full_backward_hook(bwd_h)
        in_c_prime = in_t.clone().detach().to(dev).requires_grad_(True); ng_map = None
        try:
            out_p = m_prime(in_c_prime); scr_p = torch.sigmoid(out_p).mean(); m_prime.zero_grad(); scr_p.backward()
            if acts_p is None or grads_p is None: print(f"ImageProcessor Service Err: No acts/grads NG1 {tgt_ln}.")
            else: ng_map = (torch.linalg.norm(acts_p, 2, 1, False) * torch.linalg.norm(grads_p, 2, 1, False)).squeeze(0).cpu().numpy()
        except Exception as e: print(f"ImageProcessor Service Err NG1 prime {tgt_ln}: {e}")
        finally:
            h_f.remove(); h_b.remove(); m_prime.zero_grad(); del m_prime
            if in_c_prime.grad is not None: in_c_prime.grad.zero_()
        return ng_map

    def _get_normgrad_order1_saliency_single_layer(self, m, img_b, d, tgt_ln, **kw): 
        h,w=img_b.shape[2:]
        ng_map=self._compute_normgrad_order1_analysis(m, img_b[0].unsqueeze(0), tgt_ln, d,eps=kw.get('epsilon', NORMGRAD_EPSILON_ANALYSIS),adv=kw.get('adversarial', False))
        if ng_map is None: ng_map=np.zeros((h,w))
        elif ng_map.shape!=(h,w):
            try: ng_map=cv2.resize(ng_map,(w,h),interpolation=cv2.INTER_LINEAR)
            except Exception as e: print(f"ImageProcessor Service Warn: Resize NG1 {tgt_ln} err: {e}.");ng_map=np.zeros((h,w))
        return torch.from_numpy(ng_map).float().unsqueeze(0).to(d)

    def _normalize_heatmap_tensor(self, h_t): 
        min_v,max_v=h_t.min(),h_t.max()
        return (h_t-min_v)/(max_v-min_v) if max_v-min_v>1e-8 else torch.zeros_like(h_t)

    def get_combined_normgrad_saliency_map(self, seg_model, image_tensor_batch, target_layers, status_update_fn=None, **kwargs):
        B, C, H, W = image_tensor_batch.shape
        valid_maps = []
        total_l = len(target_layers)
        for i, ln in enumerate(target_layers):
            if status_update_fn: status_update_fn(f"NormGrad Layer: {ln[:30]}... ({i+1}/{total_l})")
            s_map = self._get_normgrad_order1_saliency_single_layer(seg_model, image_tensor_batch, self.device, ln, **kwargs)
            if s_map is not None and s_map.nelement() > 0: valid_maps.append(self._normalize_heatmap_tensor(s_map.squeeze(0)))
        
        if valid_maps: 
            combined_map_tensor = torch.mean(torch.stack(valid_maps,0),0) 
            ng_h_map_np = combined_map_tensor.cpu().numpy()
            min_h_val,max_h_val = ng_h_map_np.min(),ng_h_map_np.max() 
            ng_h_map_np = (ng_h_map_np-min_h_val)/(max_h_val-min_h_val) if max_h_val-min_h_val>1e-8 else np.zeros_like(ng_h_map_np)
            return ng_h_map_np
        print("ImageProcessor Service Warn: No valid NG1 maps."); return np.zeros((H,W), dtype=np.float32)


    def get_binary_mask(self, image_tensor_batch, threshold=0.5, raw_output_tensor=None):
        if raw_output_tensor is None: # If raw output not provided, predict it
            if image_tensor_batch is None:
                print("ImageProcessor Service Error: Neither image_tensor_batch nor raw_output_tensor provided to get_binary_mask.")
                return None
            raw_output_tensor = self.model_handler.predict_segmentation(image_tensor_batch)

        binary_mask = (torch.sigmoid(raw_output_tensor) > threshold).float().squeeze().cpu().numpy().astype(np.uint8)
        return binary_mask 

    def extract_objects_from_mask(self, binary_mask_np): 
        if binary_mask_np is None or not isinstance(binary_mask_np, np.ndarray) or binary_mask_np.ndim != 2:
            print("ImageProcessor Service: Invalid binary_mask for object extraction.")
            return [], None, None 
        
        num_labels, labels_map, stats, centroids = cv2.connectedComponentsWithStats(binary_mask_np, 8, cv2.CV_32S)
        object_list = []
        total_pixels = binary_mask_np.shape[0] * binary_mask_np.shape[1]

        if num_labels <= 1: return [], None, None 

        for i in range(1, num_labels): 
            obj_info = {
                'object_id': i,
                'size_pixels': stats[i, cv2.CC_STAT_AREA],
                'relative_size': (stats[i, cv2.CC_STAT_AREA] / total_pixels) if total_pixels > 0 else 0,
                'mean_index': (centroids[i, 1], centroids[i, 0]), 
                'bbox': (stats[i, cv2.CC_STAT_LEFT], stats[i, cv2.CC_STAT_TOP], 
                         stats[i, cv2.CC_STAT_WIDTH], stats[i, cv2.CC_STAT_HEIGHT]) 
            }
            object_list.append(obj_info)
        return object_list, labels_map, stats

    def calculate_object_scores(self, object_list, labels_map, saliency_heatmap_np, 
                                img_h, img_w, 
                                config_params): 
        
        if saliency_heatmap_np is not None and labels_map is not None and labels_map.shape == saliency_heatmap_np.shape:
            for obj_item in object_list:
                obj_pixels_saliency = saliency_heatmap_np[labels_map == obj_item['object_id']]
                obj_item['mean_saliency'] = np.mean(obj_pixels_saliency) if obj_pixels_saliency.size > 0 else 0.0
        else:
            for obj_item in object_list: obj_item['mean_saliency'] = 0.0

        center_y_region = (img_h * 0.15, img_h * 0.85) 
        center_x_region = (img_w * 0.15, img_w * 0.85)
        center_of_image = np.array([img_h / 2.0, img_w / 2.0])
        
        corners = [np.array([0,0]), np.array([0,img_w]), np.array([img_h,0]), np.array([img_h,img_w])]
        max_dist_from_center = 0
        if corners: max_dist_from_center = max(np.linalg.norm(c - center_of_image) for c in corners)
        if max_dist_from_center < 1e-6: max_dist_from_center = max(img_h, img_w, 1.0)

        for obj_item in object_list:
            obj_y, obj_x = obj_item['mean_index'] 
            obj_center_np = np.array([obj_y, obj_x])
            distance = np.linalg.norm(obj_center_np - center_of_image)
            
            is_central = (center_y_region[0] <= obj_y <= center_y_region[1]) and \
                         (center_x_region[0] <= obj_x <= center_x_region[1])
            
            obj_item['importance_score'] = max(1.0 if is_central else 1.0 - min(distance / max_dist_from_center, 1.0), 0.0)

        final_scores_list = []
        for o_info in object_list:
            raw_s = o_info['relative_size']
            k_sig = config_params.get('SIGMOID_K_PARAM', DEFAULT_SIGMOID_K)
            s_thr_sig = config_params.get('SIGMOID_THRESH_PARAM', DEFAULT_SIGMOID_THRESH)
            
            penalty_s = 0.0
            if raw_s > 0:
                try:
                    adj_input = k_sig * (raw_s - s_thr_sig)
                    penalty_s = 1 / (1 + math.exp(-adj_input))
                except OverflowError: penalty_s = 0.0 if adj_input < 0 else 1.0
            penalty_s = min(max(penalty_s, 0.0), 1.0)

            w_imp = config_params.get('WEIGHT_IMPORTANCE', 0.3)
            w_sp = config_params.get('WEIGHT_SIZE_PENALTY', 0.7)
            
            ipo = (w_imp * o_info['importance_score'] + w_sp * penalty_s)
            ipo = min(max(ipo, 0.0), 1.0)
            
            final_scores_list.append({
                'object_id': int(o_info['object_id']),
                'mean_index': (float(o_info['mean_index'][0]), float(o_info['mean_index'][1])),
                'RAW_SIZE': float(raw_s),
                'PENALTY_SIZE': float(penalty_s),
                'IMPORTANCE': float(o_info['importance_score']),
                'CONFIDENT': float(o_info.get('mean_saliency', 0.0)), 
                'individual_object_penalty': float(ipo),
                'bbox': [int(b) for b in o_info['bbox']]
            })
        return final_scores_list

    def _plot_to_numpy_array(self, fig, target_size=None): 
        default_h, default_w = (self.analysis_img_size, self.analysis_img_size) if isinstance(self.analysis_img_size, int) else self.analysis_img_size

        if fig is None:
            if target_size: return np.zeros((target_size[1], target_size[0], 3), dtype=np.uint8)
            return np.zeros((default_h, default_w, 3), dtype=np.uint8)
        
        io_buf = io.BytesIO()
        
        if target_size:
            array_np_shape = (target_size[1], target_size[0], 3)
        else: 
            fig_w_px_calc, fig_h_px_calc = fig.get_size_inches() * fig.get_dpi()
            array_np_shape = (int(fig_h_px_calc), int(fig_w_px_calc), 3) if fig_w_px_calc > 0 and fig_h_px_calc > 0 else (default_h, default_w, 3)
        
        array_np = np.zeros(array_np_shape, dtype=np.uint8)

        try:
            fig.canvas.draw(); fig.savefig(io_buf, format='png', bbox_inches='tight', pad_inches=0); io_buf.seek(0)
            if io_buf.getbuffer().nbytes > 0:
                img_pil = Image.open(io_buf); img_rgb_pil = img_pil.convert('RGB')
                if target_size and (img_rgb_pil.width != target_size[0] or img_rgb_pil.height != target_size[1]):
                    img_rgb_pil = img_rgb_pil.resize(target_size, Image.LANCZOS)
                array_np_temp = np.array(img_rgb_pil, dtype=np.uint8)
                if array_np_temp.size > 0 and array_np_temp.shape[0] > 10 and array_np_temp.shape[1] > 10:
                    if target_size and array_np_temp.shape != array_np_shape:
                         print(f"ImageProcessor Service WARN plot_to_numpy_array: Resized PIL image shape {array_np_temp.shape} mismatch with target {array_np_shape}. Trying to resize CV2.")
                         array_np_temp_cv = cv2.resize(array_np_temp, (array_np_shape[1], array_np_shape[0]), interpolation=cv2.INTER_LANCZOS4)
                         if array_np_temp_cv.shape == array_np_shape:
                            array_np = array_np_temp_cv
                         else: 
                            array_np = np.array(img_rgb_pil.resize((array_np_shape[1], array_np_shape[0]), Image.LANCZOS), dtype=np.uint8)
                    else:
                        array_np = array_np_temp
                else: print("ImageProcessor Service WARN plot_to_numpy_array: Generated array too small or empty.")
            else: print("ImageProcessor Service WARN plot_to_numpy_array: io_buf empty after savefig.")
        except Exception as e: print(f"ImageProcessor Service ERROR in plot_to_numpy_array: {e}"); import traceback; traceback.print_exc()
        finally: io_buf.close(); plt.close(fig)
        return array_np

    def create_segmentation_overlay_plot(self, base_image_np, 
                                        object_labels_map, 
                                        object_scores_list, 
                                        binary_mask_fallback):
        if base_image_np is None:
            h, w = (self.analysis_img_size, self.analysis_img_size) if isinstance(self.analysis_img_size, int) else self.analysis_img_size
            return np.zeros((h, w, 3), dtype=np.uint8), "Base image not available"

        disp_img = base_image_np.copy() # Work on a copy
        if disp_img.dtype == np.uint8 : disp_img = disp_img.astype(np.float32)/255.0 
        
        is_gray = disp_img.ndim == 2 or (disp_img.ndim == 3 and disp_img.shape[2] == 1)
        disp_img_for_plot = disp_img.squeeze() if is_gray else disp_img
        cmap_disp = 'gray' if is_gray else None
        
        fig_height_px, fig_width_px = disp_img.shape[:2]

        fig, ax = plt.subplots(figsize=(fig_width_px / 100., fig_height_px / 100.), dpi=100)
        ax.imshow(disp_img_for_plot, cmap=cmap_disp); ax.axis('off')
        plot_caption = ""

        if object_scores_list: 
            n_obj = len(object_scores_list)
            cmap_tab20 = plt.get_cmap('tab20', n_obj if n_obj > 0 else 1)
            if object_labels_map is not None:
                overlay = np.zeros((*object_labels_map.shape, 4), dtype=float)
                for i, sd in enumerate(object_scores_list):
                    obj_id = sd['object_id']; rgba_color = cmap_tab20(i % cmap_tab20.N)
                    if np.any(object_labels_map == obj_id): overlay[object_labels_map == obj_id] = rgba_color
                ax.imshow(overlay, alpha=0.45)

            for i, sd in enumerate(object_scores_list):
                obj_id = sd['object_id']; rgba_color = cmap_tab20(i % cmap_tab20.N)
                x, y, w, h = sd['bbox']
                ax.add_patch(patches.Rectangle((x, y), w, h, lw=1.5, ec=rgba_color[:3], fc='none'))
                font_size = max(6, int(fig_width_px / 85))
                ax.text(x + 0.01*fig_width_px, y + 0.035*fig_height_px, f"Obj {obj_id}",
                        color='white', weight='bold', fontsize=font_size,
                        bbox=dict(boxstyle="round,pad=0.15", alpha=0.75, facecolor=rgba_color[:3], edgecolor='black', linewidth=0.3))
        else: 
            if binary_mask_fallback is not None: # binary_mask_fallback is the original binary mask
                if np.any(binary_mask_fallback): 
                    plot_caption = "Segmentation mask generated, no distinct objects scored."
                else:
                    plot_caption = "No objects detected (empty segmentation mask)."
            else: 
                plot_caption = "Segmentation mask generation failed."
        
        fig.tight_layout(pad=0)
        img_array = self._plot_to_numpy_array(fig, target_size=(fig_width_px, fig_height_px))
        if img_array.dtype != np.uint8: img_array = (np.clip(img_array,0,1) * 255).astype(np.uint8)
        return img_array, plot_caption

    def create_saliency_overlay_plot(self, base_image_np, 
                                     saliency_heatmap_np, 
                                     object_scores_list, 
                                     saliency_filter_threshold):
        if base_image_np is None:
            h, w = (self.analysis_img_size, self.analysis_img_size) if isinstance(self.analysis_img_size, int) else self.analysis_img_size
            return np.zeros((h,w, 3), dtype=np.uint8), "Base image not available"

        disp_img = base_image_np.copy()
        if disp_img.dtype == np.uint8 : disp_img = disp_img.astype(np.float32)/255.0

        is_gray = disp_img.ndim == 2 or (disp_img.ndim == 3 and disp_img.shape[2] == 1)
        disp_img_for_plot = disp_img.squeeze() if is_gray else disp_img
        cmap_disp = 'gray' if is_gray else None
        
        fig_height_px, fig_width_px = disp_img.shape[:2]

        fig, ax = plt.subplots(figsize=(fig_width_px / 100., fig_height_px / 100.), dpi=100)
        ax.imshow(disp_img_for_plot, cmap=cmap_disp); ax.axis('off')
        plot_caption = ""

        if saliency_heatmap_np is not None:
            ax.imshow(saliency_heatmap_np, cmap='jet', alpha=0.6, vmin=0, vmax=1)
            if not object_scores_list: 
                 plot_caption = "Saliency map shown. No objects scored to highlight."
        else: 
            plot_caption = "Saliency map not available." # No need for "No objects scored" if map itself isn't there.
            if not object_scores_list and plot_caption.strip().endswith('.'): # Avoid double "No objects scored"
                 plot_caption = plot_caption.strip() + " Additionally, no objects were scored."
            elif not object_scores_list:
                 plot_caption += " No objects scored."


        if object_scores_list: 
            for sd in object_scores_list:
                obj_id, bbox = sd['object_id'], sd['bbox']
                x, y, w, h = bbox
                is_above_threshold = sd.get('CONFIDENT', 0.0) > saliency_filter_threshold
                box_color = 'lime' if is_above_threshold else 'orangered'
                ax.add_patch(patches.Rectangle((x, y), w, h, lw=1.8, ec=box_color, fc='none', alpha=0.9))
                font_size = max(6, int(fig_width_px / 85))
                ax.text(x + 0.01*fig_width_px, y + 0.035*fig_height_px, f"Obj {obj_id}",
                        color='white', weight='bold', fontsize=font_size,
                        bbox=dict(boxstyle="round,pad=0.15", facecolor=box_color, alpha=0.7, edgecolor='black', linewidth=0.3))
        
        fig.tight_layout(pad=0)
        img_array = self._plot_to_numpy_array(fig, target_size=(fig_width_px, fig_height_px))
        if img_array.dtype != np.uint8: img_array = (np.clip(img_array,0,1) * 255).astype(np.uint8)
        return img_array, plot_caption.strip()