# app/routes.py
import os
import json
import traceback
import torch
import numpy as np 
import cv2         
import base64 
from flask import current_app, render_template, request, jsonify, send_file
from werkzeug.utils import secure_filename

from . import (
    data_loader, model_handler, image_processor, report_exporter 
) 
from .utils.helpers import allowed_file, np_to_base64, NumpyEncoder
from .utils.text_utils import generate_analysis_text_summary, format_analysis_for_llm_v2 
from config import NORMGRAD_TARGET_LAYERS_LIST, IMG_SIZE, TEMP_IMAGE_FOLDER, IMG_SIZE_ANALYSIS, ANALYSIS_CONFIG_GLOBAL


def run_image_quality_analysis(
        image_source_identifier_str: str, 
        analysis_tensor: torch.Tensor, 
        original_cv_image_full_res: np.ndarray, # CORRECT PARAMETER NAME
        current_analysis_config_dict: dict, 
        status_update_fn=None
    ):
    def _upd(msg):
        if status_update_fn: status_update_fn(msg)

    _upd(f"Starting analysis for {image_source_identifier_str}...")
    
    # Ensure the parameter is used correctly below
    if original_cv_image_full_res is None: # CORRECT PARAMETER NAME
        _upd("ERROR: original_cv_image_full_res is None at the start of run_image_quality_analysis.")
        return {"error": "Critical error: Full resolution original image not provided to analysis function."}

    seg_model = model_handler.get_segmentation_model()
    _upd("Predicting segmentation...")
    raw_seg_output = model_handler.predict_segmentation(analysis_tensor) 

    _upd("Generating binary mask...")
    binary_mask_analysis_res = image_processor.get_binary_mask(None, threshold=0.5, raw_output_tensor=raw_seg_output)
    if binary_mask_analysis_res is None: 
        _upd("ERROR: Failed to generate binary mask.")
        return {"error": "Binary mask generation failed"}

    _upd("Extracting objects from mask...")
    objects_analysis_res, labels_map_analysis_res, _ = image_processor.extract_objects_from_mask(binary_mask_analysis_res)
    
    _upd("Generating NormGrad saliency map...")
    normgrad_params = {
        'epsilon': current_analysis_config_dict.get('NORMGRAD_EPSILON'),
        'adversarial': current_analysis_config_dict.get('NORMGRAD_ADVERSARIAL')
    }
    saliency_map_analysis_res_np = image_processor.get_combined_normgrad_saliency_map(
        seg_model, analysis_tensor, NORMGRAD_TARGET_LAYERS_LIST, status_update_fn, **normgrad_params
    ) 

    _upd("Calculating object properties and scores...")
    analysis_h, analysis_w = analysis_tensor.shape[2], analysis_tensor.shape[3]
    object_final_scores = [] 
    if objects_analysis_res: 
        object_final_scores = image_processor.calculate_object_scores(
            objects_analysis_res, labels_map_analysis_res, saliency_map_analysis_res_np,
            analysis_h, analysis_w, current_analysis_config_dict 
        )
    else:
        _upd("No objects found after CCA for scoring.")

    q_all, p_sum_all_val, ids_all_val = 1.0, 0.0, [] 
    if object_final_scores:
        temp_p_sum_all = []
        for sd in object_final_scores: 
            ipo = sd['individual_object_penalty']
            temp_p_sum_all.append(ipo)
            ids_all_val.append(sd['object_id'])
            q_all *= (1.0 - ipo)
        q_all = min(max(q_all, 0.0), 1.0)
        p_sum_all_val = sum(temp_p_sum_all)
    num_objects_all = len(ids_all_val)

    if not object_final_scores and binary_mask_analysis_res is None: q_all = 0.0 
    elif not object_final_scores and binary_mask_analysis_res is not None: q_all = 1.0 

    q_filt, p_sum_filt_val, ids_filt_val = 1.0, 0.0, [] 
    saliency_thresh_val = current_analysis_config_dict.get('SALIENCY_FILTER_THRESHOLD')
    sfd = [o for o in object_final_scores if o.get('CONFIDENT', 0.0) > saliency_thresh_val] if object_final_scores else []
    if sfd:
        temp_p_sum_filt = []
        for sd_f in sfd: 
            ipo_f = sd_f['individual_object_penalty']
            temp_p_sum_filt.append(ipo_f)
            ids_filt_val.append(sd_f['object_id'])
            q_filt *= (1.0 - ipo_f)
        q_filt = min(max(q_filt, 0.0), 1.0)
        p_sum_filt_val = sum(temp_p_sum_filt)
    num_objects_filtered = len(ids_filt_val)
    
    if not object_final_scores and binary_mask_analysis_res is None: q_filt = 0.0
    elif not sfd and binary_mask_analysis_res is not None: q_filt = 1.0
    _upd("Quality scores calculated.")

    _upd("Generating analysis-size preview visualizations...")
    img_np_for_preview_plot = analysis_tensor.squeeze(0).cpu().permute(1,2,0).numpy()
    img_np_for_preview_plot = np.clip(img_np_for_preview_plot, 0, 1)
    if img_np_for_preview_plot.ndim == 3 and img_np_for_preview_plot.shape[-1] == 1: 
        img_np_for_preview_plot = np.concatenate([img_np_for_preview_plot]*3, axis=-1)
    
    preview_seg_plot_np, seg_caption = image_processor.create_segmentation_overlay_plot(
        img_np_for_preview_plot, labels_map_analysis_res, object_final_scores, binary_mask_analysis_res
    )
    preview_sal_plot_np, sal_caption = image_processor.create_saliency_overlay_plot(
        img_np_for_preview_plot, saliency_map_analysis_res_np, object_final_scores, saliency_thresh_val
    )
    
    # ***** THIS IS THE LINE FROM THE TRACEBACK (line 117 in your error) *****
    # Ensure original_cv_image_full_res is used here
    original_image_display_preview_np = cv2.resize(original_cv_image_full_res, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)
    # ***** END OF LINE FROM TRACEBACK *****

    if original_image_display_preview_np.dtype != np.uint8:
         if original_image_display_preview_np.max() <=1.0 : original_image_display_preview_np = (original_image_display_preview_np * 255)
         original_image_display_preview_np = np.clip(original_image_display_preview_np, 0, 255).astype(np.uint8)

    _upd("Generating full-scale visualizations for zoom/PDF...")
    orig_h_full, orig_w_full = original_cv_image_full_res.shape[:2] # Use the parameter here
    bin_m_scaled_full = None
    if binary_mask_analysis_res is not None:
        bin_m_scaled_full = cv2.resize(binary_mask_analysis_res, (orig_w_full, orig_h_full), interpolation=cv2.INTER_NEAREST)
    _, lbl_mask_scaled_full, _ = image_processor.extract_objects_from_mask(bin_m_scaled_full) if bin_m_scaled_full is not None else (None, None, None)
    obj_scores_scaled_full = []
    if object_final_scores: 
        scale_x = orig_w_full / analysis_w; scale_y = orig_h_full / analysis_h
        for score_item in object_final_scores:
            scaled_item = score_item.copy() 
            x, y, w_bbox, h_bbox = score_item['bbox'] 
            scaled_item['bbox'] = (int(x * scale_x), int(y * scale_y), int(w_bbox * scale_x), int(h_bbox * scale_y))
            my, mx = score_item['mean_index']
            scaled_item['mean_index'] = (float(my * scale_y), float(mx * scale_x))
            obj_scores_scaled_full.append(scaled_item)
    saliency_heatmap_scaled_full = None
    if saliency_map_analysis_res_np is not None:
        saliency_heatmap_scaled_full = cv2.resize(saliency_map_analysis_res_np, (orig_w_full, orig_h_full), interpolation=cv2.INTER_LINEAR)
    
    plot_base_img_full_float = original_cv_image_full_res.astype(np.float32) / 255.0 if original_cv_image_full_res.dtype == np.uint8 else original_cv_image_full_res
    
    full_seg_plot_np, _ = image_processor.create_segmentation_overlay_plot(
        plot_base_img_full_float, lbl_mask_scaled_full, obj_scores_scaled_full, bin_m_scaled_full
    )
    full_sal_plot_np, _ = image_processor.create_saliency_overlay_plot(
        plot_base_img_full_float, saliency_heatmap_scaled_full, obj_scores_scaled_full, saliency_thresh_val
    )
    full_original_img_np_uint8 = original_cv_image_full_res 
    if full_original_img_np_uint8.dtype != np.uint8: 
        if full_original_img_np_uint8.max() <=1.0: full_original_img_np_uint8 = (full_original_img_np_uint8 * 255)
        full_original_img_np_uint8 = np.clip(full_original_img_np_uint8, 0, 255).astype(np.uint8)
    _upd("Full-scale visualizations generated.")
    
    cfg_for_summary_generation = {
        'normgrad_layer_display_name':f"Comb NG O1 ({len(NORMGRAD_TARGET_LAYERS_LIST)} L)", 
        "sigmoid_k": current_analysis_config_dict.get('SIGMOID_K_PARAM'), 
        "sigmoid_thresh": current_analysis_config_dict.get('SIGMOID_THRESH_PARAM'), 
        "weight_importance": current_analysis_config_dict.get('WEIGHT_IMPORTANCE'), 
        "weight_size_penalty": current_analysis_config_dict.get('WEIGHT_SIZE_PENALTY')
    }
    technical_summary_for_ui, technical_summary_for_llm_prompt = generate_analysis_text_summary(
        image_source_identifier_str, object_final_scores, 
        q_all, p_sum_all_val, num_objects_all, 
        q_filt, p_sum_filt_val, num_objects_filtered, 
        ids_filt_val, saliency_thresh_val, 
        cfg_for_summary_generation, 
        binary_mask_analysis_res is not None
    )
    _upd("Technical summary reports (UI & LLM versions) generated.")

    results = {
        "original_image_display_preview_base64": np_to_base64(original_image_display_preview_np),
        "segmentation_overlay_preview_base64": np_to_base64(preview_seg_plot_np),
        "saliency_overlay_preview_base64": np_to_base64(preview_sal_plot_np),
        "full_scale_original_image_b64": np_to_base64(full_original_img_np_uint8),
        "full_scale_segmentation_overlay_b64": np_to_base64(full_seg_plot_np),
        "full_scale_saliency_overlay_b64": np_to_base64(full_sal_plot_np),
        "segmentation_caption": seg_caption,
        "saliency_caption": sal_caption,
        "quality_metrics": {"overall_image_quality_all":q_all, "num_objects_all":num_objects_all, 
                            "overall_image_quality_filtered":q_filt, "num_objects_filtered":num_objects_filtered},
        "text_summary_report": technical_summary_for_ui, 
        "technical_summary_for_llm": technical_summary_for_llm_prompt, 
        "saliency_filter_threshold_for_zoom": saliency_thresh_val, 
        "image_source_analyzed": image_source_identifier_str,
        "raw_binary_mask_base64": np_to_base64((binary_mask_analysis_res * 255).astype(np.uint8)) if binary_mask_analysis_res is not None else None,
        "raw_object_scores_json": json.dumps(object_final_scores, cls=NumpyEncoder) if object_final_scores else None, 
        "raw_saliency_heatmap_base64": np_to_base64((saliency_map_analysis_res_np * 255).astype(np.uint8)) if saliency_map_analysis_res_np is not None else None,
    }
    _upd("Core analysis and full-scale plot generation complete.")
    return results


@current_app.route('/analyze', methods=['POST'])
def analyze_image_route_handler(): 
    if not model_handler.is_segmentation_model_loaded(): 
        return jsonify({"error": "Segmentation model not loaded or failed to load on server."}), 503

    use_llm_from_client = request.form.get('use_llm', 'false').lower() == 'true'
    image_index_str = request.form.get('image_index')
    uploaded_file = request.files.get('image_file')
    
    analysis_tensor_for_model = None 
    original_cv_image_full_res = None # THIS IS THE VARIABLE TO BE PASSED
    current_source_identifier = "Unknown"
    current_source_identifier_for_filename = "unknown_source"
    original_image_source_ref_for_subsequent_load = None 
    is_uploaded_flag = False

    effective_analysis_config = ANALYSIS_CONFIG_GLOBAL.copy() 

    if uploaded_file and allowed_file(uploaded_file.filename):
        is_uploaded_flag = True
        try:
            filename = secure_filename(uploaded_file.filename)
            current_source_identifier = f"Uploaded: {filename}"
            current_source_identifier_for_filename = filename.rsplit('.', 1)[0] if '.' in filename else filename
            if not os.path.exists(TEMP_IMAGE_FOLDER): os.makedirs(TEMP_IMAGE_FOLDER)
            temp_original_filename = f"{secure_filename(current_source_identifier_for_filename)}_{int(time.time())}.{filename.rsplit('.',1)[-1] if '.' in filename else 'png'}"
            original_image_source_ref_for_subsequent_load = os.path.join(TEMP_IMAGE_FOLDER, temp_original_filename)
            uploaded_file.save(original_image_source_ref_for_subsequent_load)
            
            analysis_tensor_for_model, loaded_original_cv = data_loader.get_image_by_path(original_image_source_ref_for_subsequent_load)
            if analysis_tensor_for_model is None or loaded_original_cv is None:
                return jsonify({"error": "Could not process uploaded image."}), 400
            original_cv_image_full_res = loaded_original_cv 
            analysis_tensor_for_model = analysis_tensor_for_model.unsqueeze(0) 
        except Exception as e_upload:
            print(f"Flask Error processing uploaded file: {e_upload}"); traceback.print_exc()
            return jsonify({"error": f"Error processing uploaded image: {str(e_upload)}"}), 500
    elif image_index_str is not None and image_index_str.strip() != "":
        is_uploaded_flag = False
        if not data_loader.dataset_ok: return jsonify({"error": "Dataset not available for indexed analysis."}), 503
        try:
            image_index = int(image_index_str)
            if not (0 <= image_index <= data_loader.max_index):
                return jsonify({"error": f"Image index {image_index} out of bounds (0-{data_loader.max_index})."}), 400
            current_source_identifier = f"Dataset Image Index {image_index}"
            current_source_identifier_for_filename = f"dataset_idx_{image_index}"
            
            analysis_tensor_for_model, loaded_original_cv, dataset_img_path = data_loader.get_image_by_index(image_index)
            original_image_source_ref_for_subsequent_load = dataset_img_path 
            if analysis_tensor_for_model is None or loaded_original_cv is None:
                 return jsonify({"error": f"Could not load dataset image at index {image_index}."}), 400
            original_cv_image_full_res = loaded_original_cv 
            analysis_tensor_for_model = analysis_tensor_for_model.unsqueeze(0) 
        except ValueError: return jsonify({"error": "Invalid image_index format."}), 400
    else: return jsonify({"error": "No image_index or image_file provided."}), 400

    try:
        effective_analysis_config["USE_LLM_EXPLANATION"] = use_llm_from_client and model_handler.is_llm_available_and_enabled()

        analysis_results_payload = run_image_quality_analysis(
            current_source_identifier,
            analysis_tensor_for_model,
            original_cv_image_full_res, # Pass the correctly populated variable
            effective_analysis_config, 
            status_update_fn=lambda msg: print(f"ANALYSIS_LOG: {msg}")
        )
        if "error" in analysis_results_payload: 
            return jsonify(analysis_results_payload), 500

        llm_explanation_text = "LLM explanation disabled or not requested."; llm_final_status = "LLM Disabled"
        if effective_analysis_config["USE_LLM_EXPLANATION"] and model_handler.is_llm_available_and_enabled():
            print(f"Flask: Generating LLM explanation for {analysis_results_payload.get('image_source_analyzed')}...")
            try:
                technical_summary_for_llm = analysis_results_payload.get("technical_summary_for_llm")
                if technical_summary_for_llm:
                    llm_prompt = format_analysis_for_llm_v2(technical_summary_for_llm) 
                    llm_explanation_text = model_handler.generate_llm_explanation(llm_prompt)
                    llm_final_status = "LLM Success" if not llm_explanation_text.startswith("Error") else "LLM Error"
                    print(f"Flask: LLM explanation generated ({llm_final_status}).")
                else:
                    llm_explanation_text = "Technical summary for LLM was not available."
                    llm_final_status = "LLM Error - Missing Data for Prompt"
            except Exception as e_llm_route: 
                print(f"Flask ERROR during LLM prompt formatting/generation in route: {e_llm_route}"); traceback.print_exc()
                llm_explanation_text = f"Error during LLM processing: {str(e_llm_route)}"; llm_final_status = "LLM Error"
        elif effective_analysis_config["USE_LLM_EXPLANATION"] and not model_handler.is_llm_available_and_enabled():
            llm_explanation_text = "LLM was requested but is not available/loaded on the server."; llm_final_status = "LLM Not Available"
        
        final_response_data = {
            "original_image": analysis_results_payload.get("original_image_display_preview_base64"),
            "segmentation_overlay": analysis_results_payload.get("segmentation_overlay_preview_base64"),
            "saliency_overlay": analysis_results_payload.get("saliency_overlay_preview_base64"),
            "full_scale_original_b64": analysis_results_payload.get("full_scale_original_image_b64"),
            "full_scale_segmentation_b64": analysis_results_payload.get("full_scale_segmentation_overlay_b64"),
            "full_scale_saliency_b64": analysis_results_payload.get("full_scale_saliency_overlay_b64"),
            "segmentation_caption": analysis_results_payload.get("segmentation_caption"),
            "saliency_caption": analysis_results_payload.get("saliency_caption"),
            "quality_metrics": analysis_results_payload.get("quality_metrics"),
            "text_summary_report": analysis_results_payload.get("text_summary_report"), 
            "llm_explanation_text": llm_explanation_text, 
            "llm_status": llm_final_status,
            "image_source_analyzed": analysis_results_payload.get("image_source_analyzed"),
            "image_source_analyzed_for_filename": current_source_identifier_for_filename, 
            "original_image_source_ref": original_image_source_ref_for_subsequent_load, 
            "is_uploaded_original": is_uploaded_flag, 
            "raw_binary_mask_base64": analysis_results_payload.get("raw_binary_mask_base64"),
            "raw_object_scores_json": analysis_results_payload.get("raw_object_scores_json"),
            "raw_saliency_heatmap_base64": analysis_results_payload.get("raw_saliency_heatmap_base64"),
            "saliency_filter_threshold_for_zoom": analysis_results_payload.get("saliency_filter_threshold_for_zoom")
        }
        return jsonify(final_response_data)

    except Exception as e: 
        source_desc_log = current_source_identifier
        print(f"Flask: General error during analysis for {source_desc_log}: {e}"); traceback.print_exc()
        return jsonify({"error": "An internal server error occurred during analysis."}), 500

# Other routes: /get_initial_config, /export_report, /get_full_scale_plot
@current_app.route('/')
def index_route():
    return render_template('index.html', public_url=current_app.config.get('PUBLIC_URL')) 

@current_app.route('/get_initial_config', methods=['GET'])
def get_initial_config_route():
    if not data_loader or not model_handler:
        return jsonify({"error": "Server resources not initialized."}), 503
        
    return jsonify({
        "max_index": data_loader.max_index,
        "dataset_message": data_loader.dataset_message,
        "dataset_ok": data_loader.dataset_ok,
        "llm_available": model_handler.is_llm_available_and_enabled(), 
        "llm_initially_enabled": current_app.config['ANALYSIS_CONFIG_GLOBAL'].get("USE_LLM_EXPLANATION", False) and model_handler.is_llm_available_and_enabled(),
        "public_url": current_app.config.get('PUBLIC_URL') 
    })

@current_app.route('/export_report', methods=['POST'])
def export_report_route_handler():
    data_req = request.get_json()
    analysis_data_from_client = data_req.get('analysis_data') 
    if not analysis_data_from_client: 
        return jsonify({"error": "No analysis data provided for PDF export."}), 400
    
    if not report_exporter: 
        return jsonify({"error": "Report exporter not available on server."}), 503
        
    try:
        pdf_output_path, report_filename = report_exporter.generate_report_pdf(analysis_data_from_client)
        return send_file(pdf_output_path, as_attachment=True, download_name=report_filename, mimetype='application/pdf')
    except Exception as e:
        print(f"Error generating PDF from route: {e}"); traceback.print_exc()
        return jsonify({"error": f"Failed to generate PDF report: {str(e)}"}), 500

@current_app.route('/get_full_scale_plot', methods=['POST'])
def get_full_scale_plot_route_handler():
    data_req = request.get_json()
    plot_type = data_req.get('plot_type')
    original_image_ref = data_req.get('original_image_source_ref') 
    is_uploaded = data_req.get('is_uploaded_original', False)

    if not original_image_ref or not plot_type:
        return jsonify({"error": "Missing data for full scale plot."}), 400

    try:
        analysis_tensor_dummy, full_original_cv_rgb = None, None 
        if is_uploaded: 
            analysis_tensor_dummy, full_original_cv_rgb = data_loader.get_image_by_path(original_image_ref, transform_for_tensor=False)
        else: 
            analysis_tensor_dummy, full_original_cv_rgb = data_loader.get_image_by_path(original_image_ref, transform_for_tensor=False)


        if full_original_cv_rgb is None:
            return jsonify({"error": f"Could not load original image from ref: {original_image_ref}"}), 404

        if plot_type == 'original':
            img_for_modal = full_original_cv_rgb
            if img_for_modal.dtype != np.uint8:
                if img_for_modal.max() <= 1.0: img_for_modal = (img_for_modal * 255)
                img_for_modal = np.clip(img_for_modal, 0, 255).astype(np.uint8)
            return jsonify({"image_base64": np_to_base64(img_for_modal)})

        raw_bin_m_base64 = data_req.get('raw_binary_mask_base64')
        raw_obj_scores_json_str = data_req.get('raw_object_scores_json') 
        raw_saliency_heatmap_base64 = data_req.get('raw_saliency_heatmap_base64')
        saliency_filter_thresh = data_req.get('saliency_filter_threshold_for_zoom', 0.3)

        bin_m_analysis_res = None
        if raw_bin_m_base64:
            img_bytes = base64.b64decode(raw_bin_m_base64); nparr = np.frombuffer(img_bytes, np.uint8)
            bin_m_analysis_res = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)
            if bin_m_analysis_res is not None: bin_m_analysis_res = (bin_m_analysis_res > 127).astype(np.uint8)
        
        obj_scores_analysis_res = json.loads(raw_obj_scores_json_str) if raw_obj_scores_json_str else []
        
        saliency_heatmap_analysis_res = None
        if raw_saliency_heatmap_base64:
            img_bytes = base64.b64decode(raw_saliency_heatmap_base64); nparr = np.frombuffer(img_bytes, np.uint8)
            saliency_heatmap_analysis_res = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)
            if saliency_heatmap_analysis_res is not None: saliency_heatmap_analysis_res = saliency_heatmap_analysis_res / 255.0

        orig_h, orig_w = full_original_cv_rgb.shape[:2]
        analysis_h, analysis_w = current_app.config['IMG_SIZE_ANALYSIS'], current_app.config['IMG_SIZE_ANALYSIS'] 

        bin_m_scaled_full = None
        if bin_m_analysis_res is not None:
            bin_m_scaled_full = cv2.resize(bin_m_analysis_res, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)
        
        _, lbl_mask_scaled_full, _ = image_processor.extract_objects_from_mask(bin_m_scaled_full) if bin_m_scaled_full is not None else (None, None, None)
        
        obj_scores_scaled_full = []
        if obj_scores_analysis_res:
            scale_x = orig_w / analysis_w; scale_y = orig_h / analysis_h
            for score_item in obj_scores_analysis_res:
                scaled_item = score_item.copy()
                x, y, w_bbox, h_bbox = score_item['bbox'] 
                scaled_item['bbox'] = (int(x * scale_x), int(y * scale_y), int(w_bbox * scale_x), int(h_bbox * scale_y))
                my, mx = score_item['mean_index']
                scaled_item['mean_index'] = (float(my * scale_y), float(mx * scale_x))
                obj_scores_scaled_full.append(scaled_item)

        saliency_heatmap_scaled_full = None
        if saliency_heatmap_analysis_res is not None:
            saliency_heatmap_scaled_full = cv2.resize(saliency_heatmap_analysis_res, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)

        plot_base_img_float = full_original_cv_rgb.astype(np.float32) / 255.0 if full_original_cv_rgb.dtype == np.uint8 else full_original_cv_rgb

        final_plot_np = None
        if plot_type == 'segmentation':
            final_plot_np, _ = image_processor.create_segmentation_overlay_plot(
                plot_base_img_float, lbl_mask_scaled_full, obj_scores_scaled_full, bin_m_scaled_full
            )
        elif plot_type == 'saliency':
            final_plot_np, _ = image_processor.create_saliency_overlay_plot(
                plot_base_img_float, saliency_heatmap_scaled_full, obj_scores_scaled_full, saliency_filter_thresh
            )
        
        if final_plot_np is None:
            return jsonify({"error": "Failed to generate full scale plot."}), 500
        
        return jsonify({"image_base64": np_to_base64(final_plot_np)})

    except Exception as e:
        print(f"Error in get_full_scale_plot: {e}"); traceback.print_exc()
        return jsonify({"error": f"Internal server error generating full scale plot: {str(e)}"}), 500