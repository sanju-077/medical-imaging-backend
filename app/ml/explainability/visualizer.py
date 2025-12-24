"""
Explainability visualization utilities for medical imaging.
"""
import numpy as np
import cv2
from typing import Dict, Any, List, Tuple, Optional
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image, ImageDraw, ImageFont
import seaborn as sns
from io import BytesIO
import base64

from app.core.logging import medical_logger


class ExplainabilityVisualizer:
    """Generate visualizations for AI explainability in medical imaging."""
    
    def __init__(self):
        self.color_maps = {
            'viridis': plt.cm.viridis,
            'plasma': plt.cm.plasma,
            'hot': plt.cm.hot,
            'jet': plt.cm.jet,
            'red_blue': self._create_red_blue_colormap()
        }
    
    def _create_red_blue_colormap(self):
        """Create a custom red-blue colormap for medical imaging."""
        from matplotlib.colors import LinearSegmentedColormap
        colors = ['blue', 'cyan', 'white', 'yellow', 'red']
        n_bins = 256
        cmap = LinearSegmentedColormap.from_list('medical_rb', colors, N=n_bins)
        return cmap
    
    def create_heatmap_overlay(
        self,
        image: np.ndarray,
        heatmap: np.ndarray,
        alpha: float = 0.4,
        colormap: str = 'viridis',
        normalize: bool = True
    ) -> np.ndarray:
        """Create heatmap overlay on original image."""
        try:
            # Ensure image is in correct format
            if len(image.shape) == 3 and image.shape[2] == 3:
                image_rgb = image
            else:
                image_rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
            
            # Normalize heatmap if requested
            if normalize:
                heatmap_norm = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)
            else:
                heatmap_norm = heatmap
            
            # Apply colormap
            cmap = self.color_maps.get(colormap, plt.cm.viridis)
            heatmap_colored = cmap(heatmap_norm)
            
            # Convert to uint8
            if heatmap_colored.max() <= 1.0:
                heatmap_colored = (heatmap_colored * 255).astype(np.uint8)
            else:
                heatmap_colored = heatmap_colored.astype(np.uint8)
            
            # Ensure correct shape
            if len(heatmap_colored.shape) == 3:
                heatmap_rgb = (heatmap_colored[:, :, :3] * 255).astype(np.uint8)
            else:
                heatmap_rgb = cv2.applyColorMap((heatmap_norm * 255).astype(np.uint8), cv2.COLORMAP_VIRIDIS)
                heatmap_rgb = cv2.cvtColor(heatmap_rgb, cv2.COLOR_BGR2RGB)
            
            # Resize heatmap to match image if needed
            if heatmap_rgb.shape[:2] != image_rgb.shape[:2]:
                heatmap_rgb = cv2.resize(heatmap_rgb, (image_rgb.shape[1], image_rgb.shape[0]))
            
            # Create overlay
            overlay = cv2.addWeighted(image_rgb, 1 - alpha, heatmap_rgb, alpha, 0)
            
            return overlay
            
        except Exception as e:
            medical_logger.logger.error(f"Failed to create heatmap overlay: {e}")
            return image
    
    def create_comparison_visualization(
        self,
        original_image: np.ndarray,
        gradcam_heatmap: np.ndarray,
        integrated_gradients_heatmap: np.ndarray,
        method_names: List[str] = None
    ) -> np.ndarray:
        """Create side-by-side comparison of different explainability methods."""
        try:
            if method_names is None:
                method_names = ['Grad-CAM', 'Integrated Gradients']
            
            # Create figure with subplots
            fig, axes = plt.subplots(2, 2, figsize=(12, 10))
            fig.suptitle('AI Explainability Comparison', fontsize=16)
            
            # Original image
            axes[0, 0].imshow(original_image, cmap='gray')
            axes[0, 0].set_title('Original Image')
            axes[0, 0].axis('off')
            
            # Grad-CAM
            axes[0, 1].imshow(gradcam_heatmap, cmap='hot')
            axes[0, 1].set_title(method_names[0])
            axes[0, 1].axis('off')
            
            # Integrated Gradients
            axes[1, 0].imshow(integrated_gradients_heatmap, cmap='plasma')
            axes[1, 0].set_title(method_names[1])
            axes[1, 0].axis('off')
            
            # Difference map
            diff_map = np.abs(gradcam_heatmap - integrated_gradients_heatmap)
            im = axes[1, 1].imshow(diff_map, cmap='RdYlBu_r')
            axes[1, 1].set_title('Method Differences')
            axes[1, 1].axis('off')
            
            # Add colorbar for difference map
            plt.colorbar(im, ax=axes[1, 1], shrink=0.6)
            
            plt.tight_layout()
            
            # Convert to numpy array
            buffer = BytesIO()
            plt.savefig(buffer, format='png', dpi=150, bbox_inches='tight')
            buffer.seek(0)
            
            comparison_image = Image.open(buffer)
            comparison_array = np.array(comparison_image)
            
            plt.close()
            return comparison_array
            
        except Exception as e:
            medical_logger.logger.error(f"Failed to create comparison visualization: {e}")
            return original_image
    
    def create_medical_annotation(
        self,
        image: np.ndarray,
        heatmap: np.ndarray,
        anatomical_regions: Dict[str, Tuple[int, int, int, int]],
        findings: Dict[str, float],
        modality: str = 'chest_xray'
    ) -> np.ndarray:
        """Create medical annotation with anatomical regions and findings."""
        try:
            # Create PIL image for annotation
            if len(image.shape) == 3:
                pil_image = Image.fromarray(image)
            else:
                pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_GRAY2RGB))
            
            draw = ImageDraw.Draw(pil_image)
            
            # Define colors for different findings
            finding_colors = {
                'normal': (0, 255, 0),      # Green
                'abnormal': (255, 0, 0),    # Red
                'suspicious': (255, 165, 0), # Orange
                'mild': (255, 255, 0),      # Yellow
                'moderate': (255, 140, 0),  # Dark orange
                'severe': (255, 0, 0)       # Red
            }
            
            # Draw anatomical regions
            for region_name, (x, y, w, h) in anatomical_regions.items():
                # Draw rectangle for region
                color = (0, 255, 0)  # Green for normal regions
                draw.rectangle([x, y, x+w, y+h], outline=color, width=2)
                
                # Add region label
                try:
                    # Try to use a default font
                    font = ImageFont.load_default()
                except:
                    font = None
                
                draw.text((x+5, y+5), region_name, fill=color, font=font)
            
            # Add findings annotations
            y_offset = pil_image.height + 20
            draw.text((10, y_offset), "AI Findings:", fill=(0, 0, 0))
            y_offset += 20
            
            for finding, confidence in findings.items():
                # Determine color based on confidence
                if confidence > 0.8:
                    color = finding_colors['severe']
                elif confidence > 0.6:
                    color = finding_colors['moderate']
                elif confidence > 0.4:
                    color = finding_colors['mild']
                else:
                    color = finding_colors['normal']
                
                # Draw finding text
                text = f"{finding}: {confidence:.2f}"
                draw.text((10, y_offset), text, fill=color)
                y_offset += 15
            
            return np.array(pil_image)
            
        except Exception as e:
            medical_logger.logger.error(f"Failed to create medical annotation: {e}")
            return image
    
    def create_attribution_distribution_plot(
        self,
        attribution_data: np.ndarray,
        title: str = "Attribution Distribution"
    ) -> np.ndarray:
        """Create distribution plot of attribution values."""
        try:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
            
            # Histogram
            ax1.hist(attribution_data.flatten(), bins=50, alpha=0.7, color='skyblue', edgecolor='black')
            ax1.set_xlabel('Attribution Value')
            ax1.set_ylabel('Frequency')
            ax1.set_title('Attribution Distribution')
            ax1.grid(True, alpha=0.3)
            
            # Box plot
            ax2.boxplot(attribution_data.flatten())
            ax2.set_ylabel('Attribution Value')
            ax2.set_title('Attribution Statistics')
            ax2.grid(True, alpha=0.3)
            
            plt.suptitle(title)
            plt.tight_layout()
            
            # Convert to numpy array
            buffer = BytesIO()
            plt.savefig(buffer, format='png', dpi=150, bbox_inches='tight')
            buffer.seek(0)
            
            plot_image = Image.open(buffer)
            plot_array = np.array(plot_image)
            
            plt.close()
            return plot_array
            
        except Exception as e:
            medical_logger.logger.error(f"Failed to create attribution distribution plot: {e}")
            return np.array([])
    
    def create_feature_importance_visualization(
        self,
        feature_importance: Dict[str, float],
        title: str = "Feature Importance"
    ) -> np.ndarray:
        """Create feature importance bar plot."""
        try:
            features = list(feature_importance.keys())
            importance_values = list(feature_importance.values())
            
            # Create horizontal bar plot
            fig, ax = plt.subplots(figsize=(10, 6))
            
            bars = ax.barh(features, importance_values, color='lightcoral', edgecolor='black')
            
            # Color bars based on importance
            for i, (bar, value) in enumerate(zip(bars, importance_values)):
                if value > 0.7:
                    bar.set_color('red')
                elif value > 0.5:
                    bar.set_color('orange')
                elif value > 0.3:
                    bar.set_color('yellow')
                else:
                    bar.set_color('lightblue')
            
            ax.set_xlabel('Importance Score')
            ax.set_title(title)
            ax.grid(True, alpha=0.3, axis='x')
            
            # Add value labels on bars
            for i, v in enumerate(importance_values):
                ax.text(v + 0.01, i, f'{v:.3f}', va='center')
            
            plt.tight_layout()
            
            # Convert to numpy array
            buffer = BytesIO()
            plt.savefig(buffer, format='png', dpi=150, bbox_inches='tight')
            buffer.seek(0)
            
            plot_image = Image.open(buffer)
            plot_array = np.array(plot_image)
            
            plt.close()
            return plot_array
            
        except Exception as e:
            medical_logger.logger.error(f"Failed to create feature importance visualization: {e}")
            return np.array([])
    
    def create_temporal_comparison(
        self,
        current_explanation: Dict[str, Any],
        previous_explanations: List[Dict[str, Any]],
        titles: List[str] = None
    ) -> np.ndarray:
        """Create temporal comparison of explanations."""
        try:
            n_explanations = len(previous_explanations) + 1
            
            if titles is None:
                titles = ['Current'] + [f'Previous {i+1}' for i in range(len(previous_explanations))]
            
            # Create subplot grid
            cols = min(3, n_explanations)
            rows = (n_explanations + cols - 1) // cols
            
            fig, axes = plt.subplots(rows, cols, figsize=(5*cols, 4*rows))
            if n_explanations == 1:
                axes = [axes]
            elif rows == 1:
                axes = axes.reshape(1, -1)
            
            fig.suptitle('Temporal Explanation Comparison', fontsize=16)
            
            # Plot current explanation
            current_heatmap = np.array(current_explanation.get('heatmap', []))
            if current_heatmap.size > 0:
                im1 = axes[0, 0].imshow(current_heatmap, cmap='hot')
                axes[0, 0].set_title(titles[0])
                axes[0, 0].axis('off')
                plt.colorbar(im1, ax=axes[0, 0], shrink=0.6)
            
            # Plot previous explanations
            for i, prev_explanation in enumerate(previous_explanations[:n_explanations-1]):
                row = (i + 1) // cols
                col = (i + 1) % cols
                
                prev_heatmap = np.array(prev_explanation.get('heatmap', []))
                if prev_heatmap.size > 0:
                    im = axes[row, col].imshow(prev_heatmap, cmap='hot')
                    axes[row, col].set_title(titles[i+1])
                    axes[row, col].axis('off')
                    plt.colorbar(im, ax=axes[row, col], shrink=0.6)
            
            # Hide empty subplots
            for i in range(n_explanations, rows * cols):
                row = i // cols
                col = i % cols
                axes[row, col].axis('off')
            
            plt.tight_layout()
            
            # Convert to numpy array
            buffer = BytesIO()
            plt.savefig(buffer, format='png', dpi=150, bbox_inches='tight')
            buffer.seek(0)
            
            comparison_image = Image.open(buffer)
            comparison_array = np.array(comparison_image)
            
            plt.close()
            return comparison_array
            
        except Exception as e:
            medical_logger.logger.error(f"Failed to create temporal comparison: {e}")
            return np.array([])
    
    def create_interactive_heatmap(
        self,
        heatmap: np.ndarray,
        original_image: np.ndarray,
        output_path: str = None
    ) -> str:
        """Create interactive HTML heatmap visualization."""
        try:
            import plotly.graph_objects as go
            import plotly.express as px
            from plotly.subplots import make_subplots
            
            # Create subplots
            fig = make_subplots(
                rows=1, cols=2,
                subplot_titles=('Original Image', 'AI Attention Heatmap'),
                specs=[[{"type": "image"}, {"type": "heatmap"}]]
            )
            
            # Add original image
            fig.add_trace(
                go.Image(z=original_image),
                row=1, col=1
            )
            
            # Add heatmap
            fig.add_trace(
                go.Heatmap(
                    z=heatmap,
                    colorscale='Viridis',
                    showscale=True
                ),
                row=1, col=2
            )
            
            # Update layout
            fig.update_layout(
                title="Interactive AI Explainability Visualization",
                height=600,
                showlegend=False
            )
            
            # Save as HTML if path provided
            if output_path:
                fig.write_html(output_path)
                return output_path
            else:
                # Return HTML string
                return fig.to_html()
                
        except Exception as e:
            medical_logger.logger.error(f"Failed to create interactive heatmap: {e}")
            return ""
    
    def save_visualization(
        self,
        image_array: np.ndarray,
        output_path: str,
        quality: int = 95
    ) -> bool:
        """Save visualization to file."""
        try:
            if image_array.size == 0:
                return False
            
            # Convert to PIL Image
            if image_array.max() <= 1.0:
                image_array = (image_array * 255).astype(np.uint8)
            
            pil_image = Image.fromarray(image_array)
            
            # Save with specified quality
            pil_image.save(output_path, quality=quality)
            
            medical_logger.logger.info(f"Visualization saved to {output_path}")
            return True
            
        except Exception as e:
            medical_logger.logger.error(f"Failed to save visualization: {e}")
            return False
    
    def image_to_base64(self, image_array: np.ndarray) -> str:
        """Convert image array to base64 string."""
        try:
            if image_array.size == 0:
                return ""
            
            # Convert to PIL Image
            if image_array.max() <= 1.0:
                image_array = (image_array * 255).astype(np.uint8)
            
            pil_image = Image.fromarray(image_array)
            
            # Convert to base64
            buffer = BytesIO()
            pil_image.save(buffer, format='PNG')
            img_base64 = base64.b64encode(buffer.getvalue()).decode()
            
            return f"data:image/png;base64,{img_base64}"
            
        except Exception as e:
            medical_logger.logger.error(f"Failed to convert image to base64: {e}")
            return ""

