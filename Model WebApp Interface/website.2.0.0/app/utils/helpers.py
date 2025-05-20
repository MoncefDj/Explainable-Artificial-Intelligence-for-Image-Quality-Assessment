# app/utils/helpers.py
import json
import numpy as np
import base64
import io
import os 
from PIL import Image
from config import ALLOWED_EXTENSIONS 

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.int_, np.intc, np.intp, np.int8, np.int16, np.int32, np.int64, np.uint8, np.uint16, np.uint32, np.uint64)): return int(obj)
        elif isinstance(obj, (np.float_, np.float16, np.float32, np.float64)): return float(obj)
        elif isinstance(obj, (np.complex_, np.complex64, np.complex128)): return {'real': obj.real, 'imag': obj.imag}
        elif isinstance(obj, (np.ndarray,)): return obj.tolist()
        elif isinstance(obj, (np.bool_)): return bool(obj)
        return json.JSONEncoder.default(self, obj)

def sanitize_for_fpdf(text_input, is_unicode_font_active=True):
    if text_input is None:
        return ""
    
    text = str(text_input) 

    # Normalize newlines first
    text = text.replace('\r\n', '\n').replace('\r', '\n')

    if is_unicode_font_active:
        # With a Unicode font, we mainly worry about control characters or
        # characters FPDF's engine might misinterpret even if the font has the glyph.
        # FPDF's uni=True handles most of the heavy lifting for character mapping.
        # We might still replace some visually ambiguous or problematic ones if needed.
        # For now, trust the Unicode font for most characters.
        # Example: Replace zero-width space if it causes issues
        text = text.replace('\u200B', '') 
        return text # Return unicode string directly for FPDF to handle with the TTF font
    else:
        # Fallback for core fonts (approximating Windows-1252/Latin-1)
        replacements = {
            '\u2022': chr(149) if os.name == 'nt' else '*',  # Bullet
            '\u2013': '-', '\u2014': '--', # Dashes
            '\u2018': "'", '\u2019': "'", '\u201A': ',', '\u201B': "'", # Quotes
            '\u201C': '"', '\u201D': '"', '\u201E': '"', '\u201F': '"',
            '\u00AB': '<<', '\u00BB': '>>', '\u2039': '<', '\u203A': '>',
            '\u2026': '...',    # Ellipsis
            '\u00A0': ' ',      # Non-breaking space
            # Add more aggressive replacements for non-unicode path
            # Greek letters might be better represented by their names here
            '\u03A3': 'Sigma', '\u03A0': 'Pi', '\u03B1': 'alpha', # etc.
            '\u21D2': '=>', # Rightwards double arrow
        }
        for unicode_char, replacement_char in replacements.items():
            text = text.replace(unicode_char, replacement_char)
        
        try:
            return text.encode('latin-1', 'replace').decode('latin-1')
        except UnicodeEncodeError:
            return text.encode('ascii', 'replace').decode('ascii')
        except Exception as e:
            print(f"Sanitization error during fallback encoding: {e} for text snippet: {text[:50]}")
            return "".join(c if ord(c) < 128 else '?' for c in text[:100]) + "...[TRUNCATED]"


def np_to_base64(img_array_np):
    # ... (np_to_base64 function remains the same) ...
    if img_array_np is None: return None
    try:
        img_to_encode = img_array_np
        if img_array_np.dtype != np.uint8:
            if img_array_np.max() <= 1.0 and img_array_np.min() >=0.0 : img_to_encode = (img_array_np * 255)
            img_to_encode = np.clip(img_to_encode, 0, 255).astype(np.uint8)
        
        if img_to_encode.size == 0 or img_to_encode.shape[0] < 1 or img_to_encode.shape[1] < 1:
            print(f"Flask (np_to_base64): Warning - Empty or invalid shape image array: {img_to_encode.shape}")
            return None

        img_pil = Image.fromarray(img_to_encode); buffered = io.BytesIO(); img_pil.save(buffered, format="PNG"); return base64.b64encode(buffered.getvalue()).decode('utf-8')
    except Exception as e_img: 
        print(f"Flask (np_to_base64): Error converting image to base64: {e_img}")
        return None

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS