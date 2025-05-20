# app/utils/text_utils.py
import math
from config import IMG_SIZE 

def generate_analysis_text_summary(idx_or_name, obj_scores, q_all, tp_all, n_all, q_filt, tp_filt, n_filt, filt_ids, s_thr_val, cfg, bin_m_arg):
    
    def tex_ui(s): return f"${s}$" 
    def tex_display_ui(latex_code): return f"<div class=\"tex2jax_process\">$${latex_code}$$</div>"

    def tex_llm(s):
        s = s.replace(r"\tau", "tau")
        s = s.replace(r"\Sigma", "Sum")
        s = s.replace(r"\Pi", "Product")
        s = s.replace(r"\cdot", "*")
        s = s.replace(r"\Rightarrow", "=>")
        s = s.replace(r"_{\text{Imp}}", "_Imp")
        s = s.replace(r"_{\text{SizePen}}", "_SizePen")
        s = s.replace(r"_{\text{img}}", "_img")
        return s

    title = f"Image Analysis Summary (Source: {idx_or_name})"

    sigmoid_k_val = cfg.get('sigmoid_k', 0); sigmoid_thresh_val = cfg.get('sigmoid_thresh', 0)
    # Construct LaTeX strings separately to avoid f-string backslash issue
    sigmoid_latex_for_ui = f'k = {sigmoid_k_val:.1f}, \\tau = {sigmoid_thresh_val:.4f}'
    sigmoid_params_llm = f"k = {sigmoid_k_val:.1f}, tau = {sigmoid_thresh_val:.4f}"
    
    weight_imp_val = cfg.get('weight_importance', 0); weight_sp_val = cfg.get('weight_size_penalty', 0)
    scoring_weights_latex_for_ui = f'w_L = {weight_imp_val:.2f}, w_S = {weight_sp_val:.2f}'
    scoring_weights_llm = f"w_L = {weight_imp_val:.2f}, w_S = {weight_sp_val:.2f}"
    
    params_for_ui = [
        f"* Saliency Method: {cfg.get('normgrad_layer_display_name','N/A')}",
        f"* Sigmoid Parameters: {tex_ui(sigmoid_latex_for_ui)}",
        f"* Scoring Weights: {tex_ui(scoring_weights_latex_for_ui)}"
    ]
    params_for_llm_prompt = [ 
        f"* Saliency Method: {cfg.get('normgrad_layer_display_name','N/A')}",
        f"* Sigmoid Parameters: {sigmoid_params_llm}",
        f"* Scoring Weights: {scoring_weights_llm}"
    ]

    formula_latex_str = r"S_i = w_{\text{Imp}} \cdot \text{Imp}_L + w_{\text{SizePen}} \cdot \text{Pen}_S"
    formula_section_for_ui = f"""
**Scoring Methodology**:
{tex_display_ui(formula_latex_str)}
*Saliency Confidence {tex_ui('SalC')} is used for Score 2 filtering.*
"""
    formula_section_for_llm_prompt = f"""
**Scoring Methodology**:
Formula: {tex_llm(formula_latex_str)}
*Saliency Confidence (SalC) is used for Score 2 filtering.*
"""

    q_img_formula_latex = r"Q_{img} = \Pi (1 - S_i)"
    sigma_s_i_latex = r"\Sigma S_i"
    Rightarrow_latex = r"\Rightarrow"

    s1_lines_ui = [f"### Score 1: All Objects ({n_all} detected)"]
    if obj_scores:
        for sd in obj_scores:
            # Construct parts of the string separately
            rels_part = tex_ui(f'RelS={sd["RAW_SIZE"]:.3f}')
            pens_part = tex_ui(f'PenS={sd["PENALTY_SIZE"]:.3f}')
            impl_part = tex_ui(f'ImpL={sd["IMPORTANCE"]:.3f}')
            salc_part = tex_ui(f'SalC={sd["CONFIDENT"]:.3f}')
            arrow_part = tex_ui(Rightarrow_latex)
            si_part = tex_ui(f'S_i={sd["individual_object_penalty"]:.3f}')
            s1_lines_ui.append(f"  * Obj {sd['object_id']}: {rels_part}, {pens_part}, {impl_part}, {salc_part} {arrow_part} {si_part}")
    else: s1_lines_ui.append("  * No objects detected.")
    s1_lines_ui.append(f"  * Total Penalty Sum ({tex_ui(sigma_s_i_latex)} All): {tp_all:.3f}")
    s1_lines_ui.append(f"  * **Overall Quality (All, {tex_ui(q_img_formula_latex)}): {q_all if q_all is not None else 'N/A':.3f}**")
    
    s2_lines_ui = [f"### Score 2: High-Confidence Objects ({tex_ui(f'SalC > {s_thr_val}')} , {n_filt} objects)"]
    if n_filt > 0:
        s2_lines_ui.append(f"  * Filtered Object IDs: {', '.join(map(str,sorted(filt_ids)))}")
        for sd in obj_scores:
            if sd['object_id'] in filt_ids:
                rels_part = tex_ui(f'RelS={sd["RAW_SIZE"]:.3f}')
                pens_part = tex_ui(f'PenS={sd["PENALTY_SIZE"]:.3f}')
                impl_part = tex_ui(f'ImpL={sd["IMPORTANCE"]:.3f}')
                salc_part = tex_ui(f'SalC={sd["CONFIDENT"]:.3f}')
                arrow_part = tex_ui(Rightarrow_latex)
                si_part = tex_ui(f'S_i={sd["individual_object_penalty"]:.3f}')
                s2_lines_ui.append(f"  * Obj {sd['object_id']}: {rels_part}, {pens_part}, {impl_part}, {salc_part} {arrow_part} {si_part}")
    elif obj_scores: s2_lines_ui.append("  * No objects met the high-confidence saliency threshold.")
    else: s2_lines_ui.append("  * No objects detected, so no high-confidence objects to filter.")
    s2_lines_ui.append(f"  * Total Penalty Sum ({tex_ui(sigma_s_i_latex)} Filtered): {tp_filt:.3f}")
    s2_lines_ui.append(f"  * **Overall Quality (Filtered, {tex_ui(q_img_formula_latex)}): {q_filt if q_filt is not None else 'N/A':.3f}**")
    
    if not obj_scores and bin_m_arg is None: params_for_ui.append("\n**ANALYSIS FAILURE:** Mask generation error.")

    report_parts_for_ui = [f"## {title}\n\n### Configuration Parameters"]
    report_parts_for_ui.extend(params_for_ui)
    report_parts_for_ui.append("\n### Scoring Details")
    report_parts_for_ui.append(formula_section_for_ui)
    report_parts_for_ui.append("\n" + "\n".join(s1_lines_ui))
    report_parts_for_ui.append("\n" + "\n".join(s2_lines_ui))

    # Generate the version specifically for the LLM prompt (text-based math)
    report_parts_for_llm = [f"## {title}\n\n### Configuration Parameters"]
    report_parts_for_llm.extend(params_for_llm_prompt) 
    report_parts_for_llm.append("\n### Scoring Details")
    report_parts_for_llm.append(formula_section_for_llm_prompt)

    s1_lines_llm = [f"### Score 1: All Objects ({n_all} detected)"]
    if obj_scores:
        for sd in obj_scores:
            s1_lines_llm.append(f"  * Obj {sd['object_id']}: RelS={sd['RAW_SIZE']:.3f}, PenS={sd['PENALTY_SIZE']:.3f}, ImpL={sd['IMPORTANCE']:.3f}, SalC={sd['CONFIDENT']:.3f} => S_i={sd['individual_object_penalty']:.3f}")
    else: s1_lines_llm.append("  * No objects detected.")
    s1_lines_llm.append(f"  * Total Penalty Sum (Sum S_i All): {tp_all:.3f}")
    s1_lines_llm.append(f"  * **Overall Quality (All, Q_img = Product (1 - S_i)): {q_all if q_all is not None else 'N/A':.3f}**")

    s2_lines_llm = [f"### Score 2: High-Confidence Objects (SalC > {s_thr_val} , {n_filt} objects)"]
    if n_filt > 0:
        s2_lines_llm.append(f"  * Filtered Object IDs: {', '.join(map(str,sorted(filt_ids)))}")
        for sd in obj_scores:
            if sd['object_id'] in filt_ids:
                s2_lines_llm.append(f"  * Obj {sd['object_id']}: RelS={sd['RAW_SIZE']:.3f}, PenS={sd['PENALTY_SIZE']:.3f}, ImpL={sd['IMPORTANCE']:.3f}, SalC={sd['CONFIDENT']:.3f} => S_i={sd['individual_object_penalty']:.3f}")
    elif obj_scores: s2_lines_llm.append("  * No objects met the high-confidence saliency threshold.")
    else: s2_lines_llm.append("  * No objects detected, so no high-confidence objects to filter.")
    s2_lines_llm.append(f"  * Total Penalty Sum (Sum S_i Filtered): {tp_filt:.3f}")
    s2_lines_llm.append(f"  * **Overall Quality (Filtered, Q_img = Product (1 - S_i)): {q_filt if q_filt is not None else 'N/A':.3f}**")
    
    if not obj_scores and bin_m_arg is None: report_parts_for_llm.append("\n**ANALYSIS FAILURE:** Mask generation error.")

    report_parts_for_llm.append("\n" + "\n".join(s1_lines_llm))
    report_parts_for_llm.append("\n" + "\n".join(s2_lines_llm))
    
    return "\n".join(report_parts_for_ui), "\n".join(report_parts_for_llm)


def format_analysis_for_llm_v2(technical_summary_report_text_for_llm: str):
    instructions = """You are an expert in both **medical image quality assessment** (IQA) and **diagnostic radiology**, with deep experience in evaluating chest X-ray (CXR) scans. You are reviewing a technical analysis report for a CXR image that includes one or more detected objects, each described by the following parameters:

- **Score (Sᵢ)**: Impact of the object on image quality (range: 0–1). Higher values indicate greater degradation.
- **Size (RelS)**: The object’s area as a fraction of the full image (range: 0–1). Use this to estimate real-world size (e.g., if RelS ≈ 0.025 and image resolution is known, estimate ≈ 1.85 cm × 1.85 cm using standard pixel spacing of 0.117 mm).
- **Importance (ImpL)**: Relevance of the object's location to diagnostic regions (range: 0–1). A value of 1.0 means the object lies in a critical diagnostic zone (e.g., lungs, mediastinum, heart).
- **Confidence (SalC)**: AI model confidence in the object's presence (range: 0–1). A threshold of 0.3 is typically used to separate high-confidence detections.

The report also includes two image quality scores:
- **Overall Quality (All Objects)**: Computed using all detected objects.
- **Overall Quality (High-Confidence Objects)**: Computed using only objects with SalC > threshold.
---

### Generate a professional diagnostic-quality analysis report as follows:

- Write as a **single, unified report** by a senior medical imaging expert with IQA specialization.
- **Do not mention roles**, LLM behavior, or refer to “recommendations,” “here is your analysis,” etc.
- Do **not** use assistant-style phrasing or any meta-commentary.
- Use technical and clinical reasoning in clear, precise language appropriate for radiology QA or clinical review.

Specifically:
a. Don't tell me what technical analysis, just use the information gived to you to do the analysis.
b. Don't use detailed technical of each object in the analysis, use it to do the analysis
c. Do not use Recommendations or terms that seem like AI generated, and folow instructions exactly.
e. If image has not objects don't Make comparisons and Make Results Good.
f. make final decisions as in last thing saying (eg: **Results : Acceptable**/ **Good**, **Not Acceptable**)
1. Evaluate each object using its Score, Size, Importance, and Confidence.
2. Quantify each object’s contribution to image degradation (e.g., “~30% of penalty”).
3. Estimate real-world object size in cm where possible.
4. Discuss clinical implications of the object's location and severity.
5. Assess if the image is **diagnostically acceptable**.
6. Reference the difference between **All vs High-Confidence Quality scores** if they are different to assess reliability.
7. Use medical imaging terminology (e.g., cardiac silhouette, lung apex, mediastinum, hilar region).

"""

    full_prompt = f"{instructions}\n\n--- START OF ANNEX: TECHNICAL ANALYSIS ---\n\n{technical_summary_report_text_for_llm}\n\n--- END OF ANNEX: TECHNICAL ANALYSIS ---\n\nBegin Analysis:"
    
    return full_prompt