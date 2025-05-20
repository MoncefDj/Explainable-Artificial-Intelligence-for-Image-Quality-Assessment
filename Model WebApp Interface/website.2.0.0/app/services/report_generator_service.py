# app/services/report_generator_service.py
import os
import time
import base64
import io
import json
import re 
import cv2
import numpy as np
from PIL import Image 
from fpdf import FPDF 
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
import tempfile 

from app.utils.helpers import sanitize_for_fpdf 
from config import UPLOAD_FOLDER as CFG_UPLOAD_FOLDER_NAME 
from config import IMG_SIZE_ANALYSIS

# --- Font Path Definitions (relative to this file's location) ---
SERVICE_DIR = os.path.dirname(os.path.abspath(__file__))
APP_DIR_FROM_SERVICE = os.path.abspath(os.path.join(SERVICE_DIR, '..')) 
STATIC_FONTS_DIR = os.path.join(APP_DIR_FROM_SERVICE, 'static', 'fonts', 'dejavu-fonts', 'dejavu-fonts-ttf-2.37', 'ttf')

DEJAVU_SANS_PATH = os.path.join(STATIC_FONTS_DIR, "DejaVuSans.ttf")
DEJAVU_SANS_BOLD_PATH = os.path.join(STATIC_FONTS_DIR, "DejaVuSans-Bold.ttf")
DEJAVU_SANS_ITALIC_PATH = os.path.join(STATIC_FONTS_DIR, "DejaVuSans-Oblique.ttf") 
DEJAVU_SANS_BOLDITALIC_PATH = os.path.join(STATIC_FONTS_DIR, "DejaVuSans-BoldOblique.ttf") 
# --- End Font Path Definitions ---

LATEX_TO_REPLACEMENT_MAP = {
    r"\alpha": "α", r"\beta": "β", r"\gamma": "γ", r"\delta": "δ", r"\epsilon": "ε", r"\zeta": "ζ",
    r"\eta": "η", r"\theta": "θ", r"\iota": "ι", r"\kappa": "κ", r"\lambda": "λ", r"\mu": "μ",
    r"\nu": "ν", r"\xi": "ξ", r"\omicron": "ο", r"\pi": "π", r"\rho": "ρ", r"\sigma": "σ",
    r"\tau": "τ", r"\upsilon": "υ", r"\phi": "φ", r"\chi": "χ", r"\psi": "ψ", r"\omega": "ω",
    r"\varepsilon": "ε", r"\vartheta": "θ", r"\varpi": "ϖ", r"\varrho": "ϱ", r"\varsigma": "ς", r"\varphi": "φ",
    r"\Gamma": "Γ", r"\Delta": "Δ", r"\Theta": "Θ", r"\Lambda": "Λ", r"\Xi": "Ξ", r"\Pi": "Π", 
    r"\Sigma": "Σ", r"\Upsilon": "Υ", r"\Phi": "Φ", r"\Psi": "Ψ", r"\Omega": "Ω", 
    r"\cdot": "·", r"\times": "×", r"\div": "÷", r"\pm": "±", r"\mp": "∓",
    r"\leq": "≤", r"\geq": "≥", r"\neq": "≠", r"\approx": "≈", r"\equiv": "≡",
    r"\forall": "∀", r"\exists": "∃", r"\nabla": "∇", r"\partial": "∂", r"\infty": "∞",
    r"\rightarrow": "→", r"\Rightarrow": "⇒", r"\leftrightarrow": "↔", r"\Leftrightarrow": "⇔",
    r"\sum": "∑", r"\prod": "∏", r"\int": "∫",
    r"\sqrt": "sqrt", r"\frac": "/",   
    r"\text{Imp}": "Imp", r"{\text{Imp}}": "Imp",
    r"\text{SizePen}": "SizePen", r"{\text{SizePen}}": "SizePen",
    r"\text{img}": "img", r"{\text{img}}": "img",
    r"\{": "{", r"\}": "}", r"\_": "_", r"\ ": " ",
    r"\\": " " 
}

class ReportPDFServer(FPDF): # This class definition starts at column 0
    # All methods within this class are indented (e.g., by 4 spaces)
    def __init__(self, *args, latex_temp_img_dir, **kwargs): 
        super().__init__(*args, **kwargs)
        self.alias_nb_pages()
        self.set_auto_page_break(auto=True, margin=15)
        self.has_unicode_font = False
        self.is_first_page = True 
        
        self.latex_temp_img_dir = latex_temp_img_dir 
        if self.latex_temp_img_dir and not os.path.exists(self.latex_temp_img_dir):
            os.makedirs(self.latex_temp_img_dir, exist_ok=True)
        
        fonts_loaded_count = 0
        try:
            if os.path.exists(DEJAVU_SANS_PATH): self.add_font("DejaVu", "", DEJAVU_SANS_PATH, uni=True); fonts_loaded_count+=1
            else: print(f"ReportPDFServer WARNING: Font file not found: {DEJAVU_SANS_PATH}")
            if os.path.exists(DEJAVU_SANS_BOLD_PATH): self.add_font("DejaVu", "B", DEJAVU_SANS_BOLD_PATH, uni=True); fonts_loaded_count+=1
            else: print(f"ReportPDFServer WARNING: Font file not found: {DEJAVU_SANS_BOLD_PATH}")
            if os.path.exists(DEJAVU_SANS_ITALIC_PATH): self.add_font("DejaVu", "I", DEJAVU_SANS_ITALIC_PATH, uni=True); fonts_loaded_count+=1
            else: print(f"ReportPDFServer WARNING: Font file not found: {DEJAVU_SANS_ITALIC_PATH}")
            if os.path.exists(DEJAVU_SANS_BOLDITALIC_PATH): self.add_font("DejaVu", "BI", DEJAVU_SANS_BOLDITALIC_PATH, uni=True); fonts_loaded_count+=1
            else: print(f"ReportPDFServer WARNING: Font file not found: {DEJAVU_SANS_BOLDITALIC_PATH}")

            if fonts_loaded_count > 0: 
                self.set_font("DejaVu", "", 10) 
                self.has_unicode_font = True
                print(f"ReportPDFServer: DejaVuSans Unicode fonts ({fonts_loaded_count}/4 variants) processed.")
                if fonts_loaded_count < 4: print("ReportPDFServer WARNING: Not all DejaVu font variants found/loaded.")
            else: raise RuntimeError("No DejaVu fonts (regular variant at least) found. Please check paths and file existence.")
        except Exception as e:
            print(f"ReportPDFServer ERROR loading DejaVu fonts: {e}. Falling back to Arial.")
            self.set_font("Arial", "", 10) 
            self.has_unicode_font = False

    def _get_font_family(self):
        return "DejaVu" if self.has_unicode_font else "Arial"

    def header(self): 
        if self.is_first_page:
            font_family = self._get_font_family()
            self.set_font(font_family, 'B', 16)
            title_text = "IQA for X-Ray Chest and AI Analysis" 
            title_text_sanitized = sanitize_for_fpdf(title_text, self.has_unicode_font)
            title_w = self.get_string_width(title_text_sanitized) + 6 
            page_width_for_title = self.w - 2 * self.l_margin 
            self.set_x(self.l_margin + (page_width_for_title - title_w) / 2 if title_w < page_width_for_title else self.l_margin)
            self.cell(title_w if title_w < page_width_for_title else page_width_for_title, 10, title_text_sanitized, 0, 1, 'C')
            self.ln(10) 
            self.set_font(font_family, '', 10)
            self.is_first_page = False

    def footer(self):
        self.set_y(-15)
        self.set_font(self._get_font_family(), 'I', 8)
        self.cell(0, 10, sanitize_for_fpdf(f'Page {self.page_no()}/{{nb}}', self.has_unicode_font), 0, 0, 'C')

    def chapter_title(self, title_str):
        if self.get_y() + 20 > self.h - self.b_margin : 
            self.add_page()
        self.set_font(self._get_font_family(), 'B', 14)
        self.set_fill_color(200, 220, 255) 
        self.cell(0, 8, sanitize_for_fpdf(title_str, self.has_unicode_font), 0, 1, 'L', True) 
        self.ln(4)
        self.set_font(self._get_font_family(), '', 10)
        
    def _render_latex_to_image(self, latex_code, fontsize=10, dpi=150, is_display_style=False):
        if not latex_code: return None
        formatted_latex = str(latex_code).strip()
        if not (formatted_latex.startswith('$') and formatted_latex.endswith('$')):
             formatted_latex = f"${formatted_latex}$" 

        replacements = {r"\\text{Imp}": r"{\rm Imp}", r"\\text{SizePen}": r"{\rm SizePen}", r"\\text{img}": r"{\rm img}", r"\\cdot": r"\cdot", r"\\Sigma": r"\Sigma", r"\\Pi": r"\Pi", r"\\Rightarrow": r"\Rightarrow"}
        for tex_cmd, replacement in replacements.items():
            formatted_latex = formatted_latex.replace(tex_cmd, replacement)
        
        fig = None 
        try:
            fig = plt.figure(figsize=(2, 0.5), dpi=dpi) 
            ax = fig.add_subplot(111)
            text_obj = ax.text(0, 0, formatted_latex, fontsize=fontsize, ha='left', va='bottom', color='black')
            ax.axis('off') 
            fig.canvas.draw() 
            renderer = fig.canvas.get_renderer()
            bbox_disp = text_obj.get_window_extent(renderer=renderer)
            bbox_fig = bbox_disp.transformed(fig.dpi_scale_trans.inverted())

            if not self.latex_temp_img_dir or not os.path.isdir(self.latex_temp_img_dir):
                print(f"ERROR: latex_temp_img_dir ('{self.latex_temp_img_dir}') not set or not a directory.")
                if fig: plt.close(fig)
                return None
            
            temp_img_fd, temp_img_path = tempfile.mkstemp(suffix=".png", dir=self.latex_temp_img_dir)
            os.close(temp_img_fd)
            
            pad_inches = 0.02 # Adjusted padding for tighter fit
            fig.savefig(temp_img_path, bbox_inches=bbox_fig.padded(pad_inches / fig.dpi), dpi=dpi, transparent=True, pad_inches=0)
            plt.close(fig)
            fig = None
            return temp_img_path
        except Exception as e:
            print(f"Matplotlib mathtext rendering error for '{formatted_latex}': {e}")
            if fig is not None and plt.fignum_exists(fig.number): plt.close(fig)
            return None

    def _insert_formula_image(self, img_path, is_display_math=False, line_height_mm=5.5, default_font_size=10):
        if not img_path or not os.path.exists(img_path):
            self.write(line_height_mm, sanitize_for_fpdf("[Formula Img Err]", self.has_unicode_font))
            return

        try:
            with Image.open(img_path) as img_pil: width_px, height_px = img_pil.size
            dpi_render = 150 # Match DPI used in _render_latex_to_image if changed
            img_width_mm = (width_px / dpi_render) * 25.4
            img_height_mm = (height_px / dpi_render) * 25.4

            if is_display_math:
                max_width_mm = self.w - self.l_margin - self.r_margin - 2
                if img_width_mm > max_width_mm:
                    scale = max_width_mm / img_width_mm
                    img_width_mm *= scale; img_height_mm *= scale
                if self.get_y() + img_height_mm + 4 > self.h - self.b_margin: self.add_page()
                self.ln(2) 
                img_x = (self.w - img_width_mm) / 2 
                self.image(img_path, x=img_x, y=self.get_y(), w=img_width_mm, h=img_height_mm)
                self.set_y(self.get_y() + img_height_mm + 2) 
            else: 
                target_inline_height_mm = default_font_size * 0.352778 * 0.8 
                if img_height_mm > target_inline_height_mm: # Only scale down if too tall
                    scale = target_inline_height_mm / img_height_mm
                    img_height_mm *= scale; img_width_mm *= scale

                if self.get_x() + img_width_mm + 1 > self.w - self.r_margin: 
                    self.ln(line_height_mm); self.set_x(self.l_margin)
                
                image_y_pos = self.get_y() + (line_height_mm - img_height_mm) / 2.0 
                self.image(img_path, x=self.get_x(), y=image_y_pos, w=img_width_mm, h=img_height_mm)
                self.set_x(self.get_x() + img_width_mm + 0.5) 
        except Exception as e:
            print(f"Error inserting formula image {img_path}: {e}")
            self.write(line_height_mm, sanitize_for_fpdf("[Formula Ins Err]", self.has_unicode_font))
        finally:
            if os.path.exists(img_path):
                try: os.remove(img_path)
                except Exception as e_del: print(f"Error deleting temp formula image {img_path}: {e_del}")

    def write_line_with_mixed_content(self, line, default_font_size=10, line_height_mm=5.5):
        font_family = self._get_font_family()
        self.set_font(font_family, '', default_font_size)
        
        math_pattern = r'(\[Display Formula:\s*(?P<display_formula>.*?)\]|\[Math:\s*(?P<inline_formula>.*?)\])'
        style_pattern = r'(\*\*(?P<bold_text>.*?)\*\*)|(\*(?P<italic_text>.*?)\*(?!\*))'
        
        # Combine patterns: first try math, then style, then plain text
        # This order is important. (?P<plain_text>.*?) must be last and non-greedy.
        combined_pattern = re.compile(
            r'(?P<display_formula_tag>\[Display Formula:\s*(?P<display_formula_content>.*?)\])|'
            r'(?P<math_tag>\[Math:\s*(?P<math_content>.*?)\])|'
            r'(?P<bold_tag>\*\*(?P<bold_content>.*?)\*\*)|'
            r'(?P<italic_tag>\*(?P<italic_content>.*?)\*(?!\*))|'
            r'(?P<text_segment>[^\[\*]+)|'  # Plain text not starting with [ or *
            r'(?P<char_misc>[\[\*])'      # Individual [ or * if not part of patterns
        )

        last_idx = 0
        current_x = self.get_x()

        for match in combined_pattern.finditer(line):
            start, end = match.span()
            
            # Write any plain text before this match (should ideally be caught by 'text_segment')
            if start > last_idx:
                pre_text = line[last_idx:start]
                if pre_text.strip(): # Only write if not just whitespace
                    self.set_font(font_family, '', default_font_size)
                    self.write(line_height_mm, sanitize_for_fpdf(pre_text, self.has_unicode_font))
                    current_x = self.get_x()

            if self.get_y() + line_height_mm > self.h - self.b_margin: 
                self.add_page()
                current_x = self.l_margin
                self.set_x(current_x)
                self.set_font(font_family, '', default_font_size)

            if match.group('display_formula_tag'):
                formula = match.group('display_formula_content')
                img_path = self._render_latex_to_image(formula, fontsize=default_font_size + 1, is_display_style=True)
                if img_path: self._insert_formula_image(img_path, is_display_math=True, line_height_mm=line_height_mm, default_font_size=default_font_size)
                current_x = self.get_x() # Update x after image insertion
            elif match.group('math_tag'):
                formula = match.group('math_content')
                img_path = self._render_latex_to_image(formula, fontsize=default_font_size, is_display_style=False)
                if img_path: self._insert_formula_image(img_path, is_display_math=False, line_height_mm=line_height_mm, default_font_size=default_font_size)
                current_x = self.get_x()
            elif match.group('bold_tag'):
                self.set_font(font_family, 'B', default_font_size)
                self.write(line_height_mm, sanitize_for_fpdf(match.group('bold_content'), self.has_unicode_font))
                current_x = self.get_x()
            elif match.group('italic_tag'):
                self.set_font(font_family, 'I', default_font_size)
                self.write(line_height_mm, sanitize_for_fpdf(match.group('italic_content'), self.has_unicode_font))
                current_x = self.get_x()
            elif match.group('text_segment'):
                self.set_font(font_family, '', default_font_size)
                self.write(line_height_mm, sanitize_for_fpdf(match.group('text_segment'), self.has_unicode_font))
                current_x = self.get_x()
            elif match.group('char_misc'): # Write single leftover chars like '[' or '*'
                self.set_font(font_family, '', default_font_size)
                self.write(line_height_mm, sanitize_for_fpdf(match.group('char_misc'), self.has_unicode_font))
                current_x = self.get_x()
            
            last_idx = end
        
        # After processing all segments in a line, add a newline.
        # self.ln(line_height_mm) # This might add too many newlines if called from render_markdown_line which also adds ln()
        self.set_font(font_family, '', default_font_size) # Reset font


    def chapter_body_markdown(self, markdown_text, default_font_size=10):
        font_family = self._get_font_family()
        self.set_font(font_family, '', default_font_size)
        line_height_mm = 5.5 
        if markdown_text is None: markdown_text = "N/A"

        processed_text = re.sub(r'<div class="tex2jax_process">\s*\$\$(.*?)\$\$\s*</div>', 
                                lambda m: f"\n[Display Formula: {m.group(1).strip()}]\n", 
                                markdown_text, flags=re.DOTALL | re.IGNORECASE)
        processed_text = re.sub(r'(?<![\*\\])\$([^\$]+?)(?<![\*\\])\$', 
                                lambda m: f"[Math: {m.group(1).strip()}]", 
                                processed_text)
        
        html_entities = {'<': '<', '>': '>', '&': '&', '"': '"', ''': "'", ''': "'"}
        for entity, char_val in html_entities.items():
            processed_text = processed_text.replace(entity, char_val)

        lines = processed_text.split('\n')
        for line_raw in lines:
            line_content = line_raw # Keep leading/trailing spaces for block elements initially
            
            if not line_content.strip(): # Handle empty lines
                self.ln(line_height_mm / 2); continue

            if self.get_y() + line_height_mm * 1.5 > self.h - self.b_margin: 
                self.add_page(); self.set_font(font_family, '', default_font_size)

            is_block_element_handled = False # Flag to see if it's a header/list/hr
            if line_content.strip().startswith("### "):
                self.set_font(font_family, 'B', default_font_size + 1)
                self.multi_cell(0, line_height_mm, sanitize_for_fpdf(line_content[4:].strip(), self.has_unicode_font), 0, 'L')
                is_block_element_handled = True
            elif line_content.strip().startswith("## "):
                self.set_font(font_family, 'B', default_font_size + 2)
                self.multi_cell(0, line_height_mm + 1, sanitize_for_fpdf(line_content[3:].strip(), self.has_unicode_font), 0, 'L')
                is_block_element_handled = True
            elif line_content.strip().startswith("# "):
                self.set_font(font_family, 'B', default_font_size + 4)
                self.multi_cell(0, line_height_mm + 2, sanitize_for_fpdf(line_content[2:].strip(), self.has_unicode_font), 0, 'L')
                is_block_element_handled = True
            elif line_content.strip().startswith(("* ", "- ", "• ")): 
                self.set_x(self.l_margin + 5) 
                bullet = "•  " if self.has_unicode_font and font_family == "DejaVu" else "*  "
                self.set_font(font_family, '', default_font_size) 
                self.write(line_height_mm, sanitize_for_fpdf(bullet, self.has_unicode_font))
                self.write_line_with_mixed_content(line_content.lstrip('*-• ').strip(), default_font_size, line_height_mm)
                self.ln(line_height_mm) # Ensure list item gets its own line break after content
                is_block_element_handled = True
            elif line_content.strip() == "---" or line_content.strip() == "***": 
                self.line(self.l_margin, self.get_y() + line_height_mm/2, self.w - self.r_margin, self.get_y() + line_height_mm/2)
                self.ln(line_height_mm)
                is_block_element_handled = True
            
            if not is_block_element_handled and line_content.strip(): # Regular paragraph
                self.set_x(self.l_margin) 
                self.write_line_with_mixed_content(line_content.strip(), default_font_size, line_height_mm)
                # write_line_with_mixed_content should end with a newline if it processes text
                # If write_line_with_mixed_content manages its own ln() at the end for paragraphs,
                # then an additional ln() here might be too much.
                # Let's assume write_line_with_mixed_content handles a "line" and then we call ln() here.
                self.ln(line_height_mm) 
            
            self.set_font(font_family, '', default_font_size) 
        self.ln(3) 

    def add_scores_table(self, score_all_obj, num_all_obj, score_filt_obj, num_filt_obj):
        # ... (remains the same)
        current_font = self._get_font_family()
        self.set_font(current_font, 'B', 11)
        page_width_for_table = self.w - 2 * self.l_margin 
        col_width = (page_width_for_table / 2) - 1 
        
        y_before_table = self.get_y()
        self.multi_cell(col_width, 7, sanitize_for_fpdf('Overall Quality (All Objects)', self.has_unicode_font), 1, 'C', False)
        self.set_xy(self.l_margin + col_width, y_before_table) 
        self.multi_cell(col_width, 7, sanitize_for_fpdf('Quality (High-Confidence)', self.has_unicode_font), 1, 'C', False)
        self.ln(7) 

        y_before_data = self.get_y()
        self.set_font(current_font, '', 10)
        score_all_str = f"{score_all_obj:.3f} ({num_all_obj} objs)" if score_all_obj is not None else "N/A"
        score_filt_str = f"{score_filt_obj:.3f} ({num_filt_obj} objs)" if score_filt_obj is not None else "N/A"
        
        self.multi_cell(col_width, 7, sanitize_for_fpdf(score_all_str, self.has_unicode_font), 1, 'C', False)
        self.set_xy(self.l_margin + col_width, y_before_data)
        self.multi_cell(col_width, 7, sanitize_for_fpdf(score_filt_str, self.has_unicode_font), 1, 'C', False)
        self.ln(7 + 2) 

    def add_report_image(self, image_base64, title_str, temp_img_dir_for_pdf):
        # ... (remains the same) ...
        title_str_sanitized = sanitize_for_fpdf(title_str, self.has_unicode_font)
        if not image_base64:
            self.ln(5)
            self.set_font(self._get_font_family(), 'I', 10)
            self.cell(0, 10, f"[Image: {title_str_sanitized} not available]", 0, 1, 'C')
            self.ln(5)
            return

        try:
            img_data = base64.b64decode(image_base64)
            img_pil = Image.open(io.BytesIO(img_data))
            
            if not os.path.exists(temp_img_dir_for_pdf):
                os.makedirs(temp_img_dir_for_pdf, exist_ok=True)
            
            img_format = img_pil.format if img_pil.format else 'PNG'
            if img_format.upper() not in ['JPEG', 'PNG', 'GIF']: img_format = 'PNG'

            temp_img_filename = f"report_img_{time.time_ns()}.{img_format.lower()}"
            temp_img_path = os.path.join(temp_img_dir_for_pdf, temp_img_filename)
            
            save_format = img_format.upper()
            if save_format == 'JPEG':
                if img_pil.mode == 'RGBA': img_pil = img_pil.convert('RGB')
                img_pil.save(temp_img_path, format="JPEG", quality=85)
            else: 
                if img_pil.mode == 'RGBA' or img_pil.mode == 'LA': img_pil = img_pil.convert('RGB')
                img_pil.save(temp_img_path, format="PNG", optimize=True)

            img_width_pil, img_height_pil = img_pil.size
            page_usable_width_mm = self.w - 2 * self.l_margin 
            target_display_width_mm = page_usable_width_mm * (3/4) 
            
            aspect_ratio = img_height_pil / img_width_pil if img_width_pil > 0 else 1
            display_width_mm = target_display_width_mm
            display_height_mm = display_width_mm * aspect_ratio

            max_img_height_mm = (self.h - self.t_margin - self.b_margin) * 0.40 
            if display_height_mm > max_img_height_mm:
                display_height_mm = max_img_height_mm
                display_width_mm = display_height_mm / aspect_ratio if aspect_ratio > 0 else display_width_mm
            
            if display_width_mm > target_display_width_mm :
                 display_width_mm = target_display_width_mm
                 display_height_mm = display_width_mm * aspect_ratio

            required_space = display_height_mm + 10 
            if self.get_y() + required_space > (self.h - self.b_margin):
                self.add_page()

            x_pos_mm = self.l_margin + (page_usable_width_mm - display_width_mm) / 2 
            self.image(temp_img_path, x=x_pos_mm, w=display_width_mm, h=display_height_mm)
            
            try: os.remove(temp_img_path) 
            except OSError as e_remove: print(f"ReportGenerator Warning: Could not remove temp image {temp_img_path}: {e_remove}")

            self.ln(2) 
            self.set_font(self._get_font_family(), 'I', 9) 
            self.cell(0, 6, title_str_sanitized, 0, 1, 'C') 
            self.ln(8) 

        except Exception as e:
            print(f"ReportGenerator Error adding image '{title_str_sanitized}' to PDF: {e}")
            self.ln(5); self.set_font(self._get_font_family(), 'I', 10); self.cell(0, 10, f"[Could not load image: {title_str_sanitized}]", 0, 1, 'C'); self.ln(5)


class ReportGeneratorService:
    def __init__(self, final_pdf_output_folder_abs_path, image_processor_instance, data_loader_instance):
        self.final_pdf_output_folder = os.path.abspath(final_pdf_output_folder_abs_path)
        if not os.path.exists(self.final_pdf_output_folder):
            os.makedirs(self.final_pdf_output_folder, exist_ok=True)
        
        self.fpdf_report_temp_img_dir = os.path.join(self.final_pdf_output_folder, "fpdf_report_images_temp") 
        if not os.path.exists(self.fpdf_report_temp_img_dir):
            os.makedirs(self.fpdf_report_temp_img_dir, exist_ok=True)
        
        self.latex_formula_temp_img_dir = os.path.join(self.final_pdf_output_folder, "latex_formula_images_temp")
        if not os.path.exists(self.latex_formula_temp_img_dir):
            os.makedirs(self.latex_formula_temp_img_dir, exist_ok=True)

    def generate_report_pdf(self, analysis_data_from_client):
        pdf = ReportPDFServer(orientation='P', unit='mm', format='A4', latex_temp_img_dir=self.latex_formula_temp_img_dir) 
        pdf.add_page() 

        if not pdf.has_unicode_font:
            print("WARNING (ReportGeneratorService): Unicode font (DejaVuSans) not loaded. PDF text might have issues.")
        
        metrics = analysis_data_from_client.get("quality_metrics", {})
        pdf.add_scores_table(
            metrics.get("overall_image_quality_all"), metrics.get("num_objects_all",0),
            metrics.get("overall_image_quality_filtered"), metrics.get("num_objects_filtered",0)
        )
        pdf.ln(5) 

        full_original_b64 = analysis_data_from_client.get("full_scale_original_b64")
        full_segmentation_b64 = analysis_data_from_client.get("full_scale_segmentation_b64")
        full_saliency_b64 = analysis_data_from_client.get("full_scale_saliency_b64")
        
        pdf.chapter_title("Visual Analysis Outputs")
        pdf.add_report_image(full_original_b64, "Original Image", self.fpdf_report_temp_img_dir)
        pdf.add_report_image(full_segmentation_b64, "Segmentation Overlay", self.fpdf_report_temp_img_dir)
        pdf.add_report_image(full_saliency_b64, "Saliency Overlay", self.fpdf_report_temp_img_dir)
        
        def check_and_add_page(current_y, estimated_height_needed):
            if current_y + estimated_height_needed > (pdf.h - pdf.b_margin):
                pdf.add_page()
                return pdf.t_margin 
            return current_y

        current_y = pdf.get_y() 
        current_y = check_and_add_page(current_y, 30) 
        pdf.set_y(current_y) 
        pdf.chapter_title("AI Analysis:")
        llm_text = analysis_data_from_client.get('llm_explanation_text', 'AI explanation not available.')
        pdf.chapter_body_markdown(llm_text) 
        
        current_y = pdf.get_y() 
        current_y = check_and_add_page(current_y, 30) 
        pdf.set_y(current_y) 
        pdf.chapter_title("Annex: Technical Analysis")
        tech_report = analysis_data_from_client.get('text_summary_report', 'Technical summary not available.')
        pdf.chapter_body_markdown(tech_report) 
        
        report_filename = f"IQA_Report_ServerSide_{analysis_data_from_client.get('image_source_analyzed_for_filename', 'report')}_{int(time.time())}.pdf"
        pdf_output_path = os.path.join(self.final_pdf_output_folder, report_filename) 
        
        print(f"ReportGeneratorService: Attempting to save PDF to absolute path: {pdf_output_path}")
        pdf.output(pdf_output_path, 'F')
        print(f"ReportGeneratorService: PDF successfully generated at {pdf_output_path}")
        return pdf_output_path, report_filename