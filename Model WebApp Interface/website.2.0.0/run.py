# run.py
import os
from app import create_app # Import only create_app
from config import NGROK_AUTH_TOKEN 

app = create_app() # This call will set app.config['PUBLIC_URL'] if ngrok starts

if __name__ == '__main__':
    # The ngrok startup logic is now fully within create_app based on WERKZEUG_RUN_MAIN.
    # run.py's role is just to start the app and print the final status.
    
    print(f"\nFlask server starting... Access locally at http://127.0.0.1:5001")
    
    # This message block will now be more accurate for the parent process
    # WERKZEUG_RUN_MAIN is only available *after* the first run if reloader is active.
    # So this check helps determine if this is the initial master process.
    if os.environ.get("WERKZEUG_RUN_MAIN") != "true": 
        current_public_url = app.config.get("PUBLIC_URL") # Get from app.config
        if current_public_url:
            print(f"SUCCESS: ngrok tunnel should be active at: {current_public_url}")
            print("         The application will be accessible via this URL once the reloader finishes (if active).")
        elif NGROK_AUTH_TOKEN and NGROK_AUTH_TOKEN != "YOUR_NGROK_AUTH_TOKEN_HERE" and NGROK_AUTH_TOKEN.strip() != "":
            # This case means NGROK was configured, but start_ngrok_service returned None (failed)
            print("WARNING: ngrok public link was not generated or an error occurred during initial attempt.")
            print("         Check ngrok logs above. Ensure no other ngrok agents are running with your token and that the token is valid.")
        else: # NGROK_AUTH_TOKEN was not set up correctly
            print("Run.py: ngrok will not start because NGROK_AUTH_TOKEN is not configured or is the placeholder.")
            
    app.run(debug=True, host='0.0.0.0', port=5001)