# app/__init__.py
import os
from flask import Flask
from pyngrok import ngrok, conf

import config 
from .services.data_loader_service import DataLoaderService
from .services.model_handler_service import ModelHandlerService
from .services.image_processor_service import ImageProcessorService
from .services.report_generator_service import ReportGeneratorService

data_loader = None
model_handler = None
image_processor = None
report_exporter = None

def start_ngrok_service(app_config_dict): 
    auth_token = app_config_dict.get('NGROK_AUTH_TOKEN')
    current_public_url = None 
    if not auth_token or auth_token == "YOUR_NGROK_AUTH_TOKEN_HERE" or auth_token == "YOUR_REPLACED_NGROK_TOKEN" or auth_token.strip() == "": 
        print("Flask App WARNING: NGROK_AUTH_TOKEN is not properly set or is still a placeholder. Public URL via ngrok will not be available.")
        return None
    try:
        conf.get_default().auth_token = auth_token
        existing_tunnels = ngrok.get_tunnels()
        for tunnel_obj in existing_tunnels:
            if tunnel_obj.conf.get("addr") and "5001" in tunnel_obj.conf.get("addr"): 
                print(f"Flask App: Found existing ngrok tunnel: {tunnel_obj.public_url}")
                current_public_url = tunnel_obj.public_url
                break
        
        if not current_public_url: 
            for tunnel_obj in existing_tunnels:
                 if tunnel_obj.conf.get("addr") and "5001" in tunnel_obj.conf.get("addr"):
                    try: ngrok.disconnect(tunnel_obj.public_url)
                    except Exception: pass 

            tunnel = ngrok.connect(5001, bind_tls=True) 
            current_public_url = tunnel.public_url
            print(f"Flask App: * New ngrok tunnel \"{current_public_url}\" -> \"http://127.0.0.1:5001\"")
        
        return current_public_url
    except Exception as e:
        print(f"Flask App ERROR: pyngrok failed: {e}")
        if "ERR_NGROK_108" in str(e): print("Flask App HINT: ERR_NGROK_108 indicates another ngrok agent session is active. Check ngrok dashboard or kill other ngrok processes.")
        elif "ERR_NGROK_4018" in str(e): print("Flask App HINT: ERR_NGROK_4018 indicates an issue with your authtoken.")
        return None


def create_app():
    global data_loader, model_handler, image_processor, report_exporter
    
    app = Flask(__name__)
    
    app.config['UPLOAD_FOLDER'] = config.UPLOAD_FOLDER 
    app.config['TEMP_IMAGE_FOLDER'] = config.TEMP_IMAGE_FOLDER 
    app.config['NGROK_AUTH_TOKEN'] = config.NGROK_AUTH_TOKEN
    app.config['ANALYSIS_CONFIG_GLOBAL'] = config.ANALYSIS_CONFIG_GLOBAL
    app.config['NORMGRAD_TARGET_LAYERS_LIST'] = config.NORMGRAD_TARGET_LAYERS_LIST
    app.config['IMG_SIZE'] = config.IMG_SIZE
    app.config['IMG_SIZE_ANALYSIS'] = config.IMG_SIZE_ANALYSIS
    app.config['PUBLIC_URL'] = None 

    if not os.path.exists(app.config['UPLOAD_FOLDER']): 
        os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
        print(f"Flask App: Created UPLOAD_FOLDER at {app.config['UPLOAD_FOLDER']}")
    if not os.path.exists(app.config['TEMP_IMAGE_FOLDER']): 
        os.makedirs(app.config['TEMP_IMAGE_FOLDER'], exist_ok=True)

    print("Flask App: Initializing services...")
    data_loader = DataLoaderService(
        data_base_path=config.DATA_BASE_PATH, 
        csv_name=config.TEST_CSV_NAME, 
        img_size_tuple=(config.IMG_SIZE, config.IMG_SIZE)
    )
    
    model_handler = ModelHandlerService(
        segmentation_model_path=config.TRAINED_MODEL_PATH,
        llm_name=config.ANALYSIS_CONFIG_GLOBAL["LLM_MODEL_NAME"],
        llm_device_str=config.ANALYSIS_CONFIG_GLOBAL["LLM_DEVICE"],
        use_llm_flag=config.ANALYSIS_CONFIG_GLOBAL["USE_LLM_EXPLANATION"]
    )
    
    try: model_handler.get_segmentation_model()
    except Exception as e: print(f"Flask App Critical Error: Could not load segmentation model: {e}")
    try:
        if model_handler.use_llm_flag: model_handler.get_llm_instance()
    except Exception as e: print(f"Flask App Warning: Could not load LLM: {e}")

    image_processor = ImageProcessorService(
        model_handler_instance=model_handler,
        analysis_img_size=config.IMG_SIZE_ANALYSIS, 
        device_str=model_handler.device.type 
    )
    
    report_exporter = ReportGeneratorService(
        final_pdf_output_folder_abs_path=app.config['UPLOAD_FOLDER'], # Pass the absolute path
        image_processor_instance=image_processor, 
        data_loader_instance=data_loader         
    )
    print("Flask App: Services initialized.")

    if os.environ.get("WERKZEUG_RUN_MAIN") != "true":
        if app.config['NGROK_AUTH_TOKEN'] and \
           app.config['NGROK_AUTH_TOKEN'] != "YOUR_NGROK_AUTH_TOKEN_HERE" and \
           app.config['NGROK_AUTH_TOKEN'] != "YOUR_REPLACED_NGROK_TOKEN" and \
           app.config['NGROK_AUTH_TOKEN'].strip() != "":
            print("Flask App (__init__): Attempting to start ngrok (main/parent process)...")
            url_from_ngrok = start_ngrok_service(app.config) 
            app.config['PUBLIC_URL'] = url_from_ngrok 
        else:
            print("Flask App (__init__): ngrok will not start because NGROK_AUTH_TOKEN is not configured in app.config or is the placeholder.")

    with app.app_context():
        from . import routes 
    
    print("Flask App: Application factory create_app finished.")
    return app