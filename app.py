"""
Streamlit web application for SkeletAI
Interactive gender prediction from hand X-rays with Grad-CAM visualization
"""
import streamlit as st
import numpy as np
import cv2
from PIL import Image
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
import os
import io

# --- Local Imports ---
# Import from other scripts in the SourceCode directory
try:
    import config
    from gradcam import GradCAM
except ImportError as e:
    st.error(f"Error: Failed to import local modules. Ensure config.py and gradcam.py are present. {e}")
    st.stop()

# --- Page Configuration ---
st.set_page_config(
    page_title="SkeletAI - Gender Estimation",
    page_icon="🩻",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Custom CSS ---
st.markdown("""
    <style>
    .main-header {
        font-size: 3rem; color: #1E88E5; text-align: center; font-weight: bold; margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.2rem; color: #666; text-align: center; margin-bottom: 2rem;
    }
    .prediction-box {
        padding: 2rem; border-radius: 10px; margin: 1rem 0;
    }
    .male-box {
        background-color: #F54927; border-left: 5px solid #1E88E5;
    }
    .female-box {
        background-color: #F54927; border-left: 5px solid #E91E63;
    }
    </style>
""", unsafe_allow_html=True)

# --- Model & Preprocessing ---

@st.cache_resource
def load_all_models():
    """Load all trained models from the 'models' directory."""
    models = {}
    for model_name in config.MODELS_TO_COMPARE:
        model_path = os.path.join(config.MODEL_DIR, f"{model_name}_best.keras")
        if os.path.exists(model_path):
            try:
                models[model_name] = load_model(model_path)
                print(f"Successfully loaded {model_name}")
            except Exception as e:
                print(f"Error loading {model_name}: {e}")
                st.warning(f"Could not load {model_name} model.")
        else:
            print(f"Model file not found for {model_name}: {model_path}")
            
    if not models:
        st.error(f"No models found in '{config.MODEL_DIR}'. Please run train_model.py first.")
        
    return models

def preprocess_image(pil_image):
    """
    Preprocesses a single PIL image for model prediction.
    Matches the validation/test preprocessing (resize and rescale).
    """
    # Convert PIL image to RGB (for 3-channel models)
    image = pil_image.convert('RGB')
    
    # Resize to the model's expected input size
    image = image.resize(config.IMAGE_SIZE)
    
    # Convert to numpy array
    image_arr = img_to_array(image)
    
    # Apply rescaling (as done in validation)
    image_arr = image_arr * config.VALIDATION_CONFIG['rescale']
    
    # Expand dimensions to create a batch of 1
    image_batch = np.expand_dims(image_arr, axis=0)
    
    return image_batch, image_arr # Return batch for prediction, array for display

@st.cache_resource
def get_gradcam_visualizer(_model, model_name):
    """
    Caches the GradCAM visualizer.
    We pass model_name, assuming GradCAM's __init__ will handle the layer lookup.
    """
    return GradCAM(_model, model_name)

def generate_visualization(model, model_name, processed_image_batch, original_image_arr):
    """
    Generates the Grad-CAM heatmap and overlay.
    """
    try:
        # --- FIX ---
        # The GradCAM class __init__ is likely bugged and expects the model_name,
        # not the layer_name. We will pass the model_name directly and let
        # the GradCAM class handle the layer lookup from config.
        
        # OLD code:
        # layer_name = config.GRADCAM_LAYER_NAMES.get(model_name)
        # if not layer_name:
        #     st.warning(f"No Grad-CAM layer defined for {model_name} in config.")
        #     return None
        # gradcam = get_gradcam_visualizer(model, layer_name)
        
        # NEW code:
        # Pass model_name directly to work around the bug in gradcam.py
        gradcam = get_gradcam_visualizer(model, model_name) 
        
        if gradcam is None:
             st.error(f"Failed to initialize Grad-CAM for {model_name}.")
             return None
        # --- END FIX ---
        
        # Compute heatmap
        # processed_image_batch has shape (1, 224, 224, 3)
        heatmap = gradcam.compute_heatmap(processed_image_batch)
        
        # Overlay heatmap on the original (non-rescaled) image array
        # Ensure original_image_arr is 3-channel RGB and uint8
        if len(original_image_arr.shape) == 2 or original_image_arr.shape[2] == 1:
             original_image_arr_rgb = cv2.cvtColor(original_image_arr.astype(np.uint8), cv2.COLOR_GRAY2RGB)
        else:
             original_image_arr_rgb = original_image_arr.astype(np.uint8)

        # Use the overlay function from GradCAM
        superimposed_img = gradcam.overlay_heatmap(
            original_image_arr_rgb, heatmap, alpha=0.6
        )
        
        return superimposed_img
    
    except Exception as e:
        st.error(f"Error generating Grad-CAM: {e}")
        return None

# --- Main Application ---

def main():
    """Main Streamlit app execution"""
    
    # Header
    st.markdown('<div class="main-header">🩻 SkeletAI</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-header">AI-Powered Gender Estimation from Hand Radiographs</div>',
        unsafe_allow_html=True
    )
    
    # --- Sidebar ---
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # Load models
        with st.spinner("Loading models..."):
            models = load_all_models()
        
        if not models:
            st.sidebar.error("No models loaded. Please train models first.")
            st.stop()

        # Model selection
        model_choice = st.selectbox(
            "Select Model",
            models.keys(),
            help="Choose the CNN architecture for prediction"
        )
        
        st.markdown("---")
        
        # About section
        st.header("ℹ️ About")
        st.markdown(
            """
            **SkeletAI** uses deep learning to predict gender from hand X-rays.
            
            **How it works:**
            1. Upload hand X-ray image
            2. AI analyzes bone structure
            3. Predicts gender with confidence
            4. Shows what the AI "sees" (Grad-CAM)
            
            **Use Cases:**
            - Forensic identification
            - Mass disaster response
            - Archaeological studies
            """
        )
        st.markdown("---")
        st.markdown("**Version:** 1.0.0")

    # --- Main Content ---
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.header("📤 Upload X-ray Image")
        
        uploaded_file = st.file_uploader(
            "Choose a hand X-ray image",
            type=['png', 'jpg', 'jpeg'],
            help="Upload a frontal view of a hand X-ray"
        )
        
        if uploaded_file is not None:
            # Load and display original image
            image_pil = Image.open(uploaded_file)
            
            # Convert to a numpy array for display and processing
            # Use 'L' mode for grayscale display, but 'RGB' for processing
            original_display_arr = np.array(image_pil.convert('L')) 
            
            st.image(original_display_arr, caption="Uploaded X-ray", use_container_width=True, clamp=True)
            st.info(f"**Image Size:** {image_pil.size[0]} × {image_pil.size[1]} pixels")

    with col2:
        st.header("🔍 Prediction Results")
        
        if uploaded_file is not None and model_choice in models:
            # Process image and predict
            with st.spinner(f"Analyzing image with {model_choice}..."):
                
                # Get the selected model
                model = models[model_choice]
                
                # Preprocess the image
                # We pass the original PIL image
                processed_image_batch, _ = preprocess_image(image_pil)
                
                # Predict
                pred_proba = model.predict(processed_image_batch, verbose=0)[0][0]
                
                # Determine gender and confidence
                if pred_proba > 0.5:
                    gender = "Male"
                    confidence = pred_proba
                else:
                    gender = "Female"
                    confidence = 1 - pred_proba
            
            # Display prediction
            box_class = "male-box" if gender == "Male" else "female-box"
            
            # --- DEBUGGING: Add raw probability score ---
            # This helps check if the model is "stuck"
            st.text(f"Raw Model Output (Sigmoid): {pred_proba:.4f}")
            # --- END DEBUGGING ---
            
            st.markdown(f"""
                <div class="prediction-box {box_class}">
                    <h2 style="margin: 0;">Predicted Gender: {gender}</h2>
                    <h3 style="margin-top: 0.5rem; color: #666;">
                        Confidence: {confidence:.2%}
                    </h3>
                </div>
            """, unsafe_allow_html=True)
            
            # FIX: Cast numpy.float32 to standard float for st.progress
            st.progress(float(confidence))
            st.caption(f"Prediction made using **{model_choice}** model")
            
            # Grad-CAM visualization
            with st.spinner("Generating explainability heatmap..."):
                # We need the original image as a displayable array
                # Use RGB for color overlay
                original_rgb_arr = np.array(image_pil.convert('RGB'))

                superimposed_img = generate_visualization(
                    model, 
                    model_choice, 
                    processed_image_batch, 
                    original_rgb_arr
                )
            
            if superimposed_img is not None:
                st.markdown("---")
                st.subheader("🔥 Grad-CAM Visualization")
                st.caption("Areas highlighted in red/yellow influenced the model's decision")
                st.image(superimposed_img, caption="Grad-CAM Heatmap Overlay", use_container_width=True)
                st.info("""
                **Interpretation:**
                - 🔴 Red areas: Strongly influenced prediction
                - 🔵 Blue/Cool areas: Minimally influenced prediction
                """)

    # --- Footer Expander ---
    st.markdown("---")
    with st.expander("⚠️ Important Notes & Model Performance"):
        st.warning("""
        **Medical Disclaimer:**
        - This tool is for research and educational purposes only.
        - Not intended for clinical diagnosis.
        - Always consult healthcare professionals for medical decisions.
        
        **Technical Notes:**
        - Works best with clear, frontal hand X-rays.
        - Accuracy may vary with image quality.
        """)
        
        st.markdown("---")
        st.subheader("📊 Model Performance (from training logs)")
        st.text("You can fill this section with the results from your 'logs' folder.")
        # Example:
        # st.markdown("""
        # ### ResNet50 Performance
        # - **Accuracy:** 93.2%
        # - **ROC-AUC:** 0.97
        # """)

if __name__ == "__main__":
    main()

