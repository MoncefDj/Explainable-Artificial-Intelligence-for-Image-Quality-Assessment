import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as patches
import numpy as np

def show_overlay(img_np, heatmap, ax, cmap='jet', alpha=0.5, title=None, colorbar=False, vmin=0, vmax=1):
    """
    Displays a heatmap overlayed on an image on a given matplotlib axes.
    """
    if img_np.ndim == 3 and img_np.shape[-1] == 1:
        img_display = np.squeeze(img_np)
        display_cmap = 'gray'
    elif img_np.ndim == 2:
        img_display = img_np
        display_cmap = 'gray'
    else:
        img_display = img_np
        display_cmap = None
    ax.imshow(img_display, cmap=display_cmap)
    im = ax.imshow(heatmap, cmap=cmap, alpha=alpha, vmin=vmin, vmax=vmax)
    if title: ax.set_title(title, fontsize=9)
    ax.axis('off')
    if colorbar:
        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.ax.tick_params(labelsize=7)
    return im

def plot_analysis_results(
    img_tensor_display, current_image_index, object_scores_data_all,
    binary_mask, labeled_mask, normgrad_heatmap,
    overall_image_quality_all, total_penalty_sum_all, num_objects_all,
    overall_image_quality_filtered, total_penalty_sum_filtered, num_objects_filtered,
    filtered_object_ids, saliency_threshold_for_filtering, config_params
):
    """
    Plots the original image, segmentation overlay, saliency overlay, and summary text.
    """
    import matplotlib
    print("\nGenerating plots...")
    try: plt.style.use('seaborn-v0_8-whitegrid')
    except OSError:
        try: plt.style.use('seaborn-whitegrid')
        except OSError:
            try: plt.style.use('ggplot')
            except OSError: print("Warning: No specific plot style applied.")
    TITLE_FONT_SIZE=12; SUBTITLE_FONT_SIZE=10; TEXT_SUMMARY_FONT_SIZE=9.5; OBJ_ID_FONT_SIZE=9; FIG_TITLE_FONT_SIZE=14
    matplotlib.rcParams.update({'font.size':TEXT_SUMMARY_FONT_SIZE, 'axes.titlesize':TITLE_FONT_SIZE, 'axes.labelsize':SUBTITLE_FONT_SIZE, 'xtick.labelsize':TEXT_SUMMARY_FONT_SIZE-1, 'ytick.labelsize':TEXT_SUMMARY_FONT_SIZE-1, 'figure.titlesize':FIG_TITLE_FONT_SIZE, 'figure.facecolor':'white', 'axes.titleweight':'bold', 'figure.titleweight':'bold'})
    img_np = img_tensor_display.cpu().permute(1,2,0).numpy(); img_np = np.clip(img_np,0,1)
    img_np_display, cmap_img_display = (img_np.squeeze(axis=2),'gray') if img_np.shape[2]==1 else (img_np,None)
    summary_title = f"Image Analysis Summary (Index: {current_image_index})"
    param_texts = [f"Saliency Info: {config_params.get('normgrad_layer', 'N/A')}", f"Sigmoid (Size Pen.): k={config_params.get('sigmoid_k',0.0):.1f}, τ={config_params.get('sigmoid_thresh',0.0):.4f}", f"Weights: Importance={config_params.get('weight_importance',0.0):.2f}, SizePen={config_params.get('weight_size_penalty',0.0):.2f}"]
    scoring_formula_texts = ["Scoring Logic:", "  IndivObjPenalty (Sᵢ) = w_L·Imp_L + w_S·Pen_S", "  (Saliency is informational & for Score 2 filtering)"]
    score1_texts = [f"--- Score 1: All Objects ({num_objects_all}) ---"]
    if object_scores_data_all:
        for score_dict in object_scores_data_all: score1_texts.append(f"  Obj {score_dict['object_id']:<2}: RelSize={score_dict['RAW_SIZE']:.3f}, Pen_S={score_dict['PENALTY_SIZE']:.3f}, Imp_L={score_dict['IMPORTANCE']:.3f}, Sal_C={score_dict['CONFIDENT']:.3f} => Sᵢ={score_dict['individual_object_penalty']:.3f}")
    score1_texts.append(f"  Σ Sᵢ (All): {total_penalty_sum_all:.3f}"); quality_all_str = f"{overall_image_quality_all:.3f}" if overall_image_quality_all is not None else "N/A"; score1_texts.append(f"  Quality (All, Π(1-Sᵢ)): {quality_all_str}")
    score2_texts = [f"--- Score 2: Saliency > {saliency_threshold_for_filtering:.2f} ({num_objects_filtered} objs) ---"]
    if num_objects_filtered > 0:
        score2_texts.append(f"  Filtered IDs: {', '.join(map(str, sorted(filtered_object_ids)))}")
        for score_dict in object_scores_data_all:
            if score_dict['object_id'] in filtered_object_ids: score2_texts.append(f"  Obj {score_dict['object_id']:<2}: RelSize={score_dict['RAW_SIZE']:.3f}, Pen_S={score_dict['PENALTY_SIZE']:.3f}, Imp_L={score_dict['IMPORTANCE']:.3f}, Sal_C={score_dict['CONFIDENT']:.3f} => Sᵢ={score_dict['individual_object_penalty']:.3f}")
    elif object_scores_data_all : score2_texts.append("  No objects met saliency threshold.")
    score2_texts.append(f"  Σ Sᵢ (Filtered): {total_penalty_sum_filtered:.3f}"); quality_filtered_str = f"{overall_image_quality_filtered:.3f}" if overall_image_quality_filtered is not None else "N/A"; score2_texts.append(f"  Quality (Filtered, Π(1-Sᵢ)): {quality_filtered_str}")
    if not object_scores_data_all and binary_mask is None: param_texts.append("ANALYSIS FAILED: Mask generation error.")
    analysis_text_for_plot = "\n".join([summary_title, ""] + param_texts + ["-"*40] + scoring_formula_texts + ["-"*40] + score1_texts + ["-"*40] + score2_texts)
    num_image_plot_rows = 3 if normgrad_heatmap is not None else 2; image_subplot_height_inches = 4.0; total_image_height_inches = num_image_plot_rows * image_subplot_height_inches; fig_height = total_image_height_inches + 1.0; fig_width = 19
    fig = plt.figure(figsize=(fig_width, fig_height)); gs = gridspec.GridSpec(num_image_plot_rows, 2, width_ratios=[3.2,1.8], wspace=0.05, hspace=0.20, left=0.03, right=0.98, bottom=0.03, top=0.94)
    fig.suptitle(f"Foreign Object Detection Analysis (Img: {current_image_index}, Sal.Thresh: {saliency_threshold_for_filtering:.2f})", y=0.975, fontsize=matplotlib.rcParams['figure.titlesize'])
    text_props_ax1 = dict(fontsize=OBJ_ID_FONT_SIZE,color='white',weight='bold', bbox=dict(boxstyle="round,pad=0.2",alpha=0.75,edgecolor='black',linewidth=0.4)); text_props_ax2_base = dict(fontsize=OBJ_ID_FONT_SIZE,weight='bold', bbox=dict(boxstyle="round,pad=0.2",facecolor='black',alpha=0.7,edgecolor='black',linewidth=0.4))
    ax0 = fig.add_subplot(gs[0,0]); ax0.imshow(img_np_display,cmap=cmap_img_display); ax0.set_title("a) Original Image",loc='left'); ax0.axis('off')
    ax1 = fig.add_subplot(gs[1,0]); ax1.imshow(img_np_display,cmap=cmap_img_display); ax1.set_title("b) Object Segmentation",loc='left'); ax1.axis('off')
    if labeled_mask is not None and object_scores_data_all:
        num_obj_plot = len(object_scores_data_all); cmap_obj_colors = plt.get_cmap('tab10',num_obj_plot if num_obj_plot > 0 else 1); segmentation_overlay = np.zeros((*labeled_mask.shape,4),dtype=float)
        for i, score_dict in enumerate(object_scores_data_all):
            obj_id = score_dict['object_id']; color_base = cmap_obj_colors(i % cmap_obj_colors.N)
            if np.any(labeled_mask == obj_id): segmentation_overlay[labeled_mask == obj_id] = color_base
            x,y,w,h = score_dict['bbox']; rect = patches.Rectangle((x,y),w,h,linewidth=1.5,edgecolor=color_base[:3],facecolor='none'); ax1.add_patch(rect); ax1.text(x+3,y+3,f"Obj {obj_id}", **{**text_props_ax1,'bbox':{**text_props_ax1['bbox'],'facecolor':color_base[:3]}}, ha='left',va='top')
        ax1.imshow(segmentation_overlay,alpha=0.4)
    elif binary_mask is not None: ax1.imshow(binary_mask,cmap='Greys',alpha=0.5); ax1.text(0.5,0.5,"No Objects Detected",ha='center',va='center',transform=ax1.transAxes,fontsize=SUBTITLE_FONT_SIZE,color='darkgray')
    else: ax1.text(0.5,0.5,"Mask Generation Failed",ha='center',va='center',transform=ax1.transAxes,fontsize=SUBTITLE_FONT_SIZE,color='red')
    if normgrad_heatmap is not None:
        ax2 = fig.add_subplot(gs[2,0]); normgrad_heatmap_for_plot = normgrad_heatmap; ax2.imshow(img_np_display,cmap=cmap_img_display); saliency_im = ax2.imshow(normgrad_heatmap_for_plot,cmap='jet',alpha=0.65,vmin=0,vmax=1); ax2.set_title("c) Saliency & Object Filter Status",loc='left'); ax2.axis('off')
        pos_ax2 = ax2.get_position(); cbar_ax = fig.add_axes([pos_ax2.x1+0.006, pos_ax2.y0+pos_ax2.height*0.1, 0.012, pos_ax2.height*0.8]); cbar = fig.colorbar(saliency_im,cax=cbar_ax); cbar.set_label('Norm. Saliency',size=SUBTITLE_FONT_SIZE-1,labelpad=10); cbar.ax.tick_params(labelsize=TEXT_SUMMARY_FONT_SIZE-1)
        if object_scores_data_all:
            for score_dict in object_scores_data_all:
                obj_id = score_dict['object_id']; x,y,w,h = score_dict['bbox']; box_color_str = 'lime' if score_dict['CONFIDENT'] > saliency_threshold_for_filtering else 'red'; rect = patches.Rectangle((x,y),w,h,linewidth=1.8,edgecolor=box_color_str,facecolor='none',alpha=0.95); ax2.add_patch(rect); ax2.text(x+3,y+3,f"Obj {obj_id}",color=box_color_str, **text_props_ax2_base, ha='left',va='top')
    elif num_image_plot_rows > 2: ax2_placeholder = fig.add_subplot(gs[2,0]); ax2_placeholder.axis('off'); ax2_placeholder.set_title("c) Saliency Overlay (Not Available)",loc='left')
    ax_text_summary = fig.add_subplot(gs[:,1]); ax_text_summary.axis('off'); ax_text_summary.text(0.02,0.975,analysis_text_for_plot, transform=ax_text_summary.transAxes, fontsize=TEXT_SUMMARY_FONT_SIZE, verticalalignment='top',horizontalalignment='left', family='monospace',linespacing=1.33, bbox=dict(boxstyle='round,pad=0.6',fc='#F0F8FF',alpha=0.9,ec='#B0C4DE'))
    plt.show(); plt.style.use('default')
