# app/services/model_handler_service.py
import os
import torch
import torch.nn as nn
from segmentation_models_pytorch import DeepLabV3Plus
# Conditional import for GPT4All
try:
    from gpt4all import GPT4All
except ImportError:
    GPT4All = None # Will be checked against config.GPT4ALL_AVAILABLE

from config import GPT4ALL_AVAILABLE # Import the availability flag

# Segmentation Model Definition (can also be in a separate models.py if it grows)
class SegmentationModel(nn.Module):
    def __init__(self,enc="resnet50",w="imagenet",cls=1): 
        super().__init__()
        self.model=DeepLabV3Plus(encoder_name=enc,encoder_weights=w,classes=cls,activation=None)
    def forward(self,x): return self.model(x)


class ModelHandlerService:
    def __init__(self, segmentation_model_path, llm_name, llm_device_str, use_llm_flag):
        self.segmentation_model_path = segmentation_model_path
        self.llm_model_name = llm_name
        self.llm_device_str = llm_device_str
        self.use_llm_flag = use_llm_flag and GPT4ALL_AVAILABLE # Actual usage depends on availability

        self._segmentation_model = None
        self._llm_instance = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"ModelHandler Service: Using device: {self.device}")

    def get_segmentation_model(self):
        if self._segmentation_model is None:
            if os.path.exists(self.segmentation_model_path):
                print(f"ModelHandler Service: Loading segmentation model from {self.segmentation_model_path}...")
                self._segmentation_model = SegmentationModel().to(self.device)
                self._segmentation_model.load_state_dict(torch.load(self.segmentation_model_path, map_location=self.device))
                self._segmentation_model.eval()
                print("ModelHandler Service: Segmentation model loaded.")
            else:
                msg = f"ModelHandler Service ERROR: Trained segmentation model path not found: {self.segmentation_model_path}."
                print(msg)
                raise FileNotFoundError(msg)
        return self._segmentation_model

    def get_llm_instance(self):
        if self._llm_instance is None and self.use_llm_flag: # Check if LLM should be used and is available
            if GPT4All is None: # Double check if library loaded
                print("ModelHandler Service: GPT4All library not available, cannot load LLM.")
                self.use_llm_flag = False # Prevent future attempts
                return None

            print(f"ModelHandler Service: Attempting to load LLM: {self.llm_model_name} on {self.llm_device_str}")
            try:
                self._llm_instance = GPT4All(model_name=self.llm_model_name, 
                                             device=self.llm_device_str, 
                                             allow_download=True, n_ctx=8192)
                print("ModelHandler Service: LLM instance loaded successfully.")
            except Exception as e_llm:
                print(f"ModelHandler Service ERROR loading LLM: {e_llm}")
                self._llm_instance = None 
                self.use_llm_flag = False # Disable on failure
        return self._llm_instance

    def is_segmentation_model_loaded(self):
        try: return self.get_segmentation_model() is not None
        except FileNotFoundError: return False
            
    def is_llm_available_and_enabled(self):
        if not self.use_llm_flag: return False
        return self.get_llm_instance() is not None

    def predict_segmentation(self, image_tensor_batch):
        model = self.get_segmentation_model()
        model.eval()
        with torch.no_grad():
            return model(image_tensor_batch.to(self.device))

    def generate_llm_explanation(self, prompt_text):
        llm = self.get_llm_instance()
        if not llm: return "LLM not available or failed to load."
        try:
            with llm.chat_session():
                explanation = llm.generate(prompt=prompt_text, max_tokens=8192, temp=0.1)
            return explanation
        except Exception as e:
            print(f"ModelHandler Service Error during LLM generation: {e}")
            import traceback
            traceback.print_exc()
            return f"Error during LLM generation: {str(e)}"