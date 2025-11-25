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
    pass # Allow standalone use

class GradCAM:
    """
    Generate Grad-CAM heatmaps for CNN classification model predictions.
    """
    def __init__(self, model, model_name):
        self.model = model
        self.model_name = model_name
        
        # 1. FIND THE TARGET LAYER
        self.layer_name = self._find_target_layer()

        # 2. CREATE THE GRADIENT MODEL
        try:
            self.grad_model = keras.models.Model(
                inputs=[self.model.input],
                outputs=[
                    self.model.get_layer(self.layer_name).output,
                    self.model.output
                ]
            )
        except Exception as e:
            raise ValueError(f"Grad-CAM Error: Could not create gradient model. "
                             f"Target layer '{self.layer_name}' caused error: {e}")

    def _find_target_layer(self):
        """
        Robustly finds the last CONVOLUTIONAL layer.
        """
        # 1. Get the ideal name from config
        desired_name = config.GRADCAM_LAYER_NAMES.get(self.model_name, "")
        
        # 2. Get all layer names
        all_layers = [l.name for l in self.model.layers]
        
        # 3. STRATEGY A: Exact Match
        if desired_name in all_layers:
            return desired_name

        # 4. STRATEGY B: Prefix Match (Handles 'block5_conv4_1' etc.)
        # Search backwards to find the deepest match
        if desired_name:
            for name in reversed(all_layers):
                if name.startswith(desired_name):
                    print(f"[GradCAM] Found variant of {desired_name}: {name}")
                    return name

        # 5. STRATEGY C: Automatic Detection of Last Conv Layer
        # If config name fails, just grab the last 4D layer (Conv2D)
        print(f"[GradCAM] Warning: Config layer '{desired_name}' not found. Auto-detecting...")
        
        for layer in reversed(self.model.layers):
            # Check if it's a Conv2D layer (4D output: batch, h, w, channels)
            if len(layer.output.shape) == 4 and 'conv' in layer.name.lower():
                print(f"[GradCAM] Auto-detected last conv layer: {layer.name}")
                return layer.name

        raise ValueError(f"Could not find any suitable convolutional layer in {self.model_name}. "
                         f"Layers checked: {all_layers[-5:]}")

    def compute_heatmap(self, image):
        if len(image.shape) == 3:
            image = np.expand_dims(image, axis=0)
            
        with tf.GradientTape() as tape:
            conv_outputs, predictions = self.grad_model(image)
            loss = predictions[0] 
        
        grads = tape.gradient(loss, conv_outputs)
        
        # Guard against None gradients (happens if layer is disconnected)
        if grads is None:
            # Fallback: Just return empty heatmap to prevent crash
            print("[GradCAM] Error: Gradients were None. Is the layer trainable?")
            return np.zeros((image.shape[1], image.shape[2]))

        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        conv_outputs = conv_outputs[0]
        heatmap = tf.einsum('ijc,c->ij', conv_outputs, pooled_grads)
        
        heatmap = heatmap.numpy()
        
        # Normalize
        heatmap = np.maximum(heatmap, 0)
        if np.max(heatmap) > 0:
            heatmap /= np.max(heatmap)
            
        return heatmap
    
    def overlay_heatmap(self, original_image_rgb, heatmap, alpha=0.5):
        # Resize heatmap to match image size
        heatmap_resized = cv2.resize(heatmap, (original_image_rgb.shape[1], original_image_rgb.shape[0]))
        
        # Convert heatmap to RGB
        heatmap_rgb = np.uint8(255 * heatmap_resized)
        heatmap_rgb = cv2.applyColorMap(heatmap_rgb, cv2.COLORMAP_JET)
        
        if original_image_rgb.max() <= 1.0:
            original_image_rgb = np.uint8(255 * original_image_rgb)
        
        original_image_bgr = cv2.cvtColor(original_image_rgb, cv2.COLOR_RGB2BGR)
        superimposed = cv2.addWeighted(original_image_bgr, 1 - alpha, heatmap_rgb, alpha, 0)
        return cv2.cvtColor(superimposed, cv2.COLOR_BGR2RGB)