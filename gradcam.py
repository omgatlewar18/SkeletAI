"""
Grad-CAM (Gradient-weighted Class Activation Mapping) implementation
for explainable AI visualization in the SkeletAI project.
"""
import numpy as np
import tensorflow as tf
from tensorflow import keras
import cv2
import matplotlib.pyplot as plt
import os

# Import settings from the central configuration file
try:
    import config
except ImportError:
    print("Error: config.py not found. Please ensure it's in the same directory.")
    exit()

class GradCAM:
    """
    Generate Grad-CAM heatmaps for CNN classification model predictions.
    """
    def __init__(self, model, model_name):
        """
        Args:
            model: Trained Keras classification model.
            model_name: Name of the model (e.g., 'ResNet50') to look up
                        the correct layer from config.py.
        """
        self.model = model
        
        # Get the target layer name from config
        if model_name not in config.GRADCAM_LAYER_NAMES:
            raise ValueError(f"Model '{model_name}' not found in config.GRADCAM_LAYER_NAMES.")
        self.layer_name = config.GRADCAM_LAYER_NAMES[model_name]

        # Create the gradient model
        try:
            self.grad_model = keras.models.Model(
                inputs=[self.model.input],
                outputs=[
                    self.model.get_layer(self.layer_name).output,
                    self.model.output
                ]
            )
        except ValueError as e:
            print(f"Error creating Grad-CAM model: {e}")
            print(f"Could not find layer '{self.layer_name}' in model '{model_name}'.")
            print("Please verify layer names in config.py.")
            exit(1)
    
    def compute_heatmap(self, image):
        """
        Compute Grad-CAM heatmap for the classification output.
        
        Args:
            image: Input image (preprocessed, ready for model).
            
        Returns:
            heatmap: Grad-CAM heatmap (normalized to [0, 1]).
        """
        if len(image.shape) == 3:
            image = np.expand_dims(image, axis=0)
        
        # Compute gradients
        with tf.GradientTape() as tape:
            conv_outputs, predictions = self.grad_model(image)
            # We want the gradient of the final output neuron (sigmoid)
            # This works for binary_crossentropy
            loss = predictions[0] 
        
        grads = tape.gradient(loss, conv_outputs)
        
        # Global average pooling of gradients
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        
        # Weight feature maps by gradients
        conv_outputs = conv_outputs[0]
        heatmap = tf.einsum('ijc,c->ij', conv_outputs, pooled_grads)
        
        # Convert to numpy
        heatmap = heatmap.numpy()
        
        # Normalize heatmap: Apply ReLU (keep positive values)
        heatmap = np.maximum(heatmap, 0)
        if heatmap.max() > 0:
            heatmap /= heatmap.max()
        
        return heatmap
    
    def overlay_heatmap(self, original_image_rgb, heatmap, alpha=0.5):
        """
        Overlay heatmap on the original RGB image.
        """
        # Resize heatmap to match image size
        heatmap_resized = cv2.resize(heatmap, (original_image_rgb.shape[1], original_image_rgb.shape[0]))
        
        # Convert heatmap to RGB
        heatmap_rgb = np.uint8(255 * heatmap_resized)
        heatmap_rgb = cv2.applyColorMap(heatmap_rgb, cv2.COLORMAP_JET)
        
        # Ensure original image is 8-bit
        if original_image_rgb.max() <= 1.0:
            original_image_rgb = np.uint8(255 * original_image_rgb)
        
        # Convert original image to BGR for cv2 overlay if it's RGB
        original_image_bgr = cv2.cvtColor(original_image_rgb, cv2.COLOR_RGB2BGR)

        # Superimpose heatmap
        superimposed = cv2.addWeighted(original_image_bgr, 1 - alpha, heatmap_rgb, alpha, 0)
        
        # Convert back to RGB for matplotlib display
        superimposed_rgb = cv2.cvtColor(superimposed, cv2.COLOR_BGR2RGB)
        
        return superimposed_rgb
    
    def visualize(self, preprocessed_image, original_image_rgb, save_path=None):
        """
        Generate complete Grad-CAM visualization.
        
        Args:
            preprocessed_image: Image ready for model input (e.g., resized, rescaled)
            original_image_rgb: Original image (RGB) for display
            save_path: (Optional) Path to save the plot.
        """
        # 1. Get prediction
        pred_prob = self.model.predict(np.expand_dims(preprocessed_image, axis=0))[0][0]
        pred_index = 1 if pred_prob > 0.5 else 0
        pred_class_name = config.CLASS_NAMES[pred_index]
        pred_confidence = pred_prob if pred_index == 1 else 1 - pred_prob
        
        # 2. Compute heatmap
        heatmap = self.compute_heatmap(preprocessed_image)
        
        # 3. Create visualization
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        # Original image
        axes[0].imshow(original_image_rgb)
        axes[0].set_title('Original X-ray', fontsize=14)
        axes[0].axis('off')
        
        # Heatmap
        axes[1].imshow(heatmap, cmap='jet')
        axes[1].set_title('Grad-CAM Heatmap', fontsize=14)
        axes[1].axis('off')
        
        # Superimposed
        superimposed = self.overlay_heatmap(original_image_rgb, heatmap)
        axes[2].imshow(superimposed)
        title = f'Predicted: {pred_class_name} ({pred_confidence*100:.1f}% Conf.)'
        axes[2].set_title(title, fontsize=14)
        axes[2].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Grad-CAM visualization saved to: {save_path}")
        else:
            plt.show()
        
        plt.close()
        
        return heatmap, superimposed

# --- Helper function for testing ---
def load_and_preprocess_image(img_path, target_size):
    """
    Loads an image for display and preprocesses it for the model.
    """
    # 1. Load original image (for display)
    original_img = cv2.imread(img_path)
    if original_img is None:
        raise FileNotFoundError(f"Image not found at {img_path}")
    original_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)
    
    # 2. Load and preprocess for model input
    # We use tf.keras utils for consistency with data_preprocessing.py
    img = tf.keras.utils.load_img(img_path, target_size=target_size, color_mode="rgb")
    img_array = tf.keras.utils.img_to_array(img)
    img_array_rescaled = img_array / 255.0 # Apply the same rescaling as in data_preprocessing
    
    return img_array_rescaled, original_img

# --- Example usage ---
if __name__ == "__main__":
    
    # --- FIX: Added missing import ---
    from tensorflow.keras.models import load_model # type: ignore
    # ---------------------------------
    
    MODEL_NAME_TO_TEST = config.DEFAULT_MODEL
    MODEL_PATH = os.path.join(config.MODEL_DIR, f"{MODEL_NAME_TO_TEST}_best.keras")
    
    # 1. Find a test image
    TEST_IMAGE_DIR = os.path.join(config.DATA_DIR, 'test', 'male') # Test with 'male'
    try:
        test_image_name = os.listdir(TEST_IMAGE_DIR)[0]
        TEST_IMAGE_PATH = os.path.join(TEST_IMAGE_DIR, test_image_name)
        print(f"Test image found: {TEST_IMAGE_PATH}")
    except Exception as e:
        print(f"Error finding test image: {e}")
        print(f"Please ensure '{TEST_IMAGE_DIR}' contains images.")
        exit()

    # 2. Load the trained model
    try:
        model = load_model(MODEL_PATH)
        print(f"Successfully loaded model: {MODEL_PATH}")
    except Exception as e:
        print(f"Error loading model: {e}")
        print(f"Ensure you have successfully run 'train_model.py' for {MODEL_NAME_TO_TEST}.")
        exit()

    # 3. Load and preprocess the image
    try:
        processed_img, original_img = load_and_preprocess_image(
            TEST_IMAGE_PATH, 
            config.IMAGE_SIZE
        )
    except FileNotFoundError as e:
        print(e)
        exit()
        
    # 4. Initialize GradCAM
    gradcam = GradCAM(model, model_name=MODEL_NAME_TO_TEST)
    
    # 5. Generate and save visualization
    save_path = os.path.join(config.LOG_DIR, f"{MODEL_NAME_TO_TEST}_gradcam_example.png")
    gradcam.visualize(
        processed_img,
        original_img,
        save_path=save_path
    )

