import traceback

def format_analysis_for_llm(
    image_index, config_params, object_scores_data_all, num_objects_all,
    object_ids_all, overall_image_quality_all, num_objects_filtered,
    filtered_object_ids, overall_image_quality_filtered,
    saliency_threshold_for_filtering, binary_mask
):
    """
    Formats the analysis results into a prompt for the LLM.
    """
    prompt_lines = [f"LLM Analysis Request: Chest X-ray Image Quality Assessment Report (Image Index: {image_index})", "Please provide a clear, concise, explanatory, and educational summary...", "...The analysis identifies foreign objects...","...A Saliency Score...","\n--- Analysis Configuration (For Reference) ---",f"  - Importance Weight (w_L): {config_params.get('weight_importance',0.0):.2f}", f"  - Size Penalty Weight (w_S): {config_params.get('weight_size_penalty',0.0):.2f}", f"  - Individual Object Penalty (Sᵢ) Formula: Sᵢ = w_L * LocationImportance + w_S * SizePenalty", f"  - Overall Image Quality (Q_img) Formula: Q_img = Π (1 - Sᵢ)"]
    if not object_scores_data_all: prompt_lines.extend(["\n--- Object Detection Summary ---", "No foreign objects were detected..."])
    else:
        prompt_lines.append("\n--- Individual Detected Object Details ---")
        for sd in object_scores_data_all: prompt_lines.append(f"  Object ID {sd['object_id']}: IndividualPenalty(Sᵢ)={sd['individual_object_penalty']:.3f} (Factors: SizePenalty={sd['PENALTY_SIZE']:.3f}, LocationImportance={sd['IMPORTANCE']:.3f}). SaliencyScore(Confidence)={sd['CONFIDENT']:.3f}.")
    def get_quality_category(score):
        if score is None: return "Not Assessed";
        if score > 0.8: return "Good";
        if score > 0.5: return "Acceptable/Borderline";
        return "Poor"
    quality_all_str_llm = f"{overall_image_quality_all:.3f}" if overall_image_quality_all is not None else "N/A"; category_all = get_quality_category(overall_image_quality_all)
    prompt_lines.extend(["\n\n--- IMAGE QUALITY ASSESSMENT REPORT ---", "\n## Overall Summary of Image Quality", f"This section summarizes the automated assessment of the chest X-ray image (Index: {image_index})...", f"\n## Score 1: Quality Based on All Detected Objects ({num_objects_all} objects)", f"  - Calculated Quality Score (Q_img_all): {quality_all_str_llm}", f"  - Interpretation: This score suggests the image quality is '{category_all}'."])
    if num_objects_all == 0 and binary_mask is not None: prompt_lines.append("    - No objects were detected, leading to a high quality score...")
    elif num_objects_all > 0: prompt_lines.append(f"    - This score is based on the combined impact of all {num_objects_all} detected object(s).")
    quality_filtered_str_llm = f"{overall_image_quality_filtered:.3f}" if overall_image_quality_filtered is not None else "N/A"; category_filtered = get_quality_category(overall_image_quality_filtered)
    prompt_lines.extend([f"\n## Score 2: Quality Based on High-Confidence Objects ({num_objects_filtered} objects)", f"  - (Objects filtered by Saliency Score / Model Confidence > {saliency_threshold_for_filtering:.2f})", f"  - Calculated Quality Score (Q_img_filtered): {quality_filtered_str_llm}", f"  - Interpretation: When considering only objects detected with higher confidence, the image quality is assessed as '{category_filtered}'."])
    if num_objects_filtered == 0 and num_objects_all > 0: prompt_lines.append("    - No objects met the high-confidence threshold...")
    elif num_objects_filtered == 0 and num_objects_all == 0: prompt_lines.append("    - No objects were detected at all.")
    prompt_lines.append("\n## Comparison and Detailed Insights")
    if overall_image_quality_all is not None and overall_image_quality_filtered is not None:
        significant_diff_threshold = 0.1; comparison_detail = ""
        if abs(overall_image_quality_all - overall_image_quality_filtered) > significant_diff_threshold:
            comparison = "significantly different"
            if overall_image_quality_filtered > overall_image_quality_all: comparison_detail = "Score 2 (high-confidence objects) is notably higher..."
            else: comparison_detail = "Score 2 (high-confidence objects) is lower or similar to Score 1..."
        else: comparison = "similar"; comparison_detail = "The two scores are similar..."
        prompt_lines.extend([f"The two quality scores ({quality_all_str_llm} vs. {quality_filtered_str_llm}) are {comparison}.", f"  - {comparison_detail}"])
    if object_scores_data_all:
        prompt_lines.append("\n### Explanation of Individual Object Penalties (Sᵢ):")
        for sd in object_scores_data_all:
            s_i = sd['individual_object_penalty']; obj_id = sd['object_id']; factors = []
            if sd['IMPORTANCE'] >= 0.8: factors.append(f"its critical location (Importance={sd['IMPORTANCE']:.2f})")
            elif sd['IMPORTANCE'] <= 0.3: factors.append(f"its less critical location (Importance={sd['IMPORTANCE']:.2f})")
            if sd['PENALTY_SIZE'] >= 0.5: factors.append(f"its large relative size (SizePenalty={sd['PENALTY_SIZE']:.2f})")
            elif sd['PENALTY_SIZE'] <= 0.1 and sd['RAW_SIZE'] > 0: factors.append(f"its small relative size (SizePenalty={sd['PENALTY_SIZE']:.2f})")
            explanation_s_i = ""
            if s_i >= 0.7: explanation_s_i = f"Object {obj_id} has a very high penalty (Sᵢ={s_i:.3f}). This is primarily due to " + (" and ".join(factors) + "." if factors else "its combined size and location impact.")
            elif s_i >= 0.4: explanation_s_i = f"Object {obj_id} has a notable penalty (Sᵢ={s_i:.3f}). This is influenced by " + (" and ".join(factors) + "." if factors else "its combined size and location impact.")
            elif s_i > 0: explanation_s_i = f"Object {obj_id} has a relatively low penalty (Sᵢ={s_i:.3f}), indicating it's considered less impactful due to " + (" or ".join(factors) + "." if factors else "its combined size and location.")
            if explanation_s_i: prompt_lines.append(f"  - {explanation_s_i} Its Saliency Score (model confidence) is {sd['CONFIDENT']:.3f}.")
    prompt_lines.append("\n## Recommendations (Based on Automated Analysis)")
    manual_review_needed = False; problematic_objects_summary = []
    if overall_image_quality_all is not None:
        if overall_image_quality_all <= 0.5: prompt_lines.append("  - Manual review is strongly suggested..."); manual_review_needed = True
        elif overall_image_quality_all <= 0.8: prompt_lines.append("  - Manual review may be warranted..."); manual_review_needed = True
    if object_scores_data_all:
        high_penalty_objects = [obj for obj in object_scores_data_all if obj['individual_object_penalty'] >= 0.6]
        high_saliency_high_penalty = [obj for obj in high_penalty_objects if obj['object_id'] in filtered_object_ids]
        if high_saliency_high_penalty: problematic_objects_summary.append(f"Object(s) {', '.join([str(o['object_id']) for o in high_saliency_high_penalty])} are particularly noteworthy...")
        elif high_penalty_objects and not manual_review_needed: problematic_objects_summary.append(f"Consider focusing on Object(s) {', '.join([str(o['object_id']) for o in high_penalty_objects])} which have high penalty...")
    if problematic_objects_summary:
        for summary_item in problematic_objects_summary: prompt_lines.append(f"  - {summary_item}")
    elif not manual_review_needed and (overall_image_quality_all is None or overall_image_quality_all > 0.8): prompt_lines.append("  - No specific objects flagged as highly problematic...")
    prompt_lines.extend(["\nNote: This is an automated analysis...", "\n--- End of Report ---"])
    return "\n".join(prompt_lines)

def run_llm_analysis(llm_model, prompt, max_tokens, temp):
    """
    Runs the LLM analysis using the provided prompt and returns the response.
    """
    try:
        with llm_model.chat_session():
            response = llm_model.generate(prompt=prompt, max_tokens=max_tokens, temp=temp)
        return response
    except Exception as e:
        print(f"Error during LLM explanation: {e}")
        traceback.print_exc()
        return None
