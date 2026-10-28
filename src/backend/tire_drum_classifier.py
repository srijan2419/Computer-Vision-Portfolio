"""
Tire Drum Classification System
Uses trained YOLOv8 model to detect and classify tire drum states.
Classes: empty_drum, in_process, yellow_tire
"""

import os
import sys
from pathlib import Path

# Force CPU mode - set BEFORE importing torch/ultralytics
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ['FORCE_CPU'] = '1'

import cv2
import torch
from ultralytics import YOLO

# Force unbuffered output for Windows
sys.stdout.reconfigure(line_buffering=True) if hasattr(sys.stdout, 'reconfigure') else None

# Disable CUDA explicitly in PyTorch
torch.cuda.is_available = lambda: False


class TireDrumClassifier:
    """Classifies tire drum images using trained YOLOv8 model."""
    
    def __init__(self, model_path="YOLO/best.pt"):
        """
        Initialize the tire drum classifier.
        
        Args:
            model_path (str): Path to the trained YOLOv8 model weights
        """
        self.model_path = Path(model_path)
        self.class_names = ['empty_drum', 'in_process', 'yellow_tire']
        self.model = None
        
        # Load model
        self._load_model()
        
    def _load_model(self):
        """Load the YOLOv8 model from weights file."""
        if not self.model_path.exists():
            raise FileNotFoundError(
                f"Model weights not found at: {self.model_path}\n"
                f"Please ensure the trained model file exists."
            )
        
        print(f"Loading model from: {self.model_path}")
        try:
            # Suppress ultralytics verbose output during loading
            import warnings
            warnings.filterwarnings('ignore')
            
            self.model = YOLO(str(self.model_path))
            self.model.overrides['verbose'] = False  # Reduce verbosity
            
            print(f"✓ Model loaded successfully!")
            print(f"  Classes: {self.class_names}")
        except Exception as e:
            raise RuntimeError(f"Failed to load model: {e}")
    
    def predict(self, image_path, conf_threshold=0.25, save_results=True):
        """
        Predict tire drum state from an image.
        
        Args:
            image_path (str): Path to the image file
            conf_threshold (float): Confidence threshold for detections (0-1)
            save_results (bool): Whether to save annotated result image
            
        Returns:
            dict: Dictionary containing prediction results
        """
        image_path = Path(image_path)
        
        # Verify image exists
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")
        
        print(f"\n{'='*60}")
        print(f"Analyzing: {image_path.name}")
        print(f"{'='*60}")
        print("Running inference (this may take a few seconds)...")
        sys.stdout.flush()
        
        try:
            # Run inference with minimal verbosity, force CPU to avoid GPU issues
            results = self.model(
                source=str(image_path),
                conf=conf_threshold,
                save=save_results,
                project="results",
                name="predictions",
                exist_ok=True,
                verbose=False,
                device='cpu',  # Force CPU to avoid potential GPU/CUDA issues
                stream=False
            )
            
            print("✓ Inference complete!")
            sys.stdout.flush()
            
        except Exception as e:
            import traceback
            print(f"✗ Inference failed: {e}")
            print("Traceback:")
            traceback.print_exc()
            sys.stdout.flush()
            raise
        
        # Parse results
        result = results[0]
        detections = []
        
        if len(result.boxes) > 0:
            for box in result.boxes:
                class_id = int(box.cls[0])
                confidence = float(box.conf[0])
                bbox = box.xyxy[0].tolist()  # [x1, y1, x2, y2]
                
                detection = {
                    'class': self.class_names[class_id],
                    'class_id': class_id,
                    'confidence': confidence,
                    'bbox': bbox
                }
                detections.append(detection)
        
        # Prepare summary
        summary = {
            'image_path': str(image_path),
            'total_detections': len(detections),
            'detections': detections,
            'image_size': result.orig_shape
        }
        
        # Display results
        self._display_results(summary)
        
        # Save annotated image path if available
        if save_results and len(results) > 0:
            summary['annotated_image'] = str(Path("results/predictions") / image_path.name)
            print(f"\n✓ Annotated image saved to: {summary['annotated_image']}", flush=True)
        
        sys.stdout.flush()
        return summary
    
    def _display_results(self, summary):
        """
        Display prediction results in a readable format.
        
        Args:
            summary (dict): Prediction summary dictionary
        """
        print(f"\nResults:", flush=True)
        print(f"  Image Size: {summary['image_size'][1]}x{summary['image_size'][0]}", flush=True)
        print(f"  Total Detections: {summary['total_detections']}", flush=True)
        
        if summary['total_detections'] > 0:
            print(f"\nDetected Objects:", flush=True)
            print(f"  {'#':<4} {'Class':<15} {'Confidence':<12} {'Bounding Box (x1, y1, x2, y2)'}", flush=True)
            print(f"  {'-'*70}", flush=True)
            
            for idx, det in enumerate(summary['detections'], 1):
                bbox_str = f"({det['bbox'][0]:.1f}, {det['bbox'][1]:.1f}, {det['bbox'][2]:.1f}, {det['bbox'][3]:.1f})"
                print(f"  {idx:<4} {det['class']:<15} {det['confidence']:.2%}        {bbox_str}", flush=True)
            
            # Summary by class
            print(f"\nSummary by Class:", flush=True)
            class_counts = {}
            for det in summary['detections']:
                class_name = det['class']
                class_counts[class_name] = class_counts.get(class_name, 0) + 1
            
            for class_name, count in sorted(class_counts.items()):
                print(f"  - {class_name}: {count}", flush=True)
        else:
            print(f"  No objects detected above confidence threshold.", flush=True)
    
    def batch_predict(self, image_folder, conf_threshold=0.25):
        """
        Run predictions on all images in a folder.
        
        Args:
            image_folder (str): Path to folder containing images
            conf_threshold (float): Confidence threshold for detections
            
        Returns:
            list: List of prediction summaries for each image
        """
        image_folder = Path(image_folder)
        
        if not image_folder.exists():
            raise FileNotFoundError(f"Folder not found: {image_folder}")
        
        # Get all image files
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
        image_files = []
        for ext in image_extensions:
            image_files.extend(image_folder.glob(f"*{ext}"))
        
        if not image_files:
            print(f"No images found in: {image_folder}")
            return []
        
        print(f"\nFound {len(image_files)} images to process...")
        
        # Process each image
        all_results = []
        for idx, image_path in enumerate(image_files, 1):
            print(f"\n[{idx}/{len(image_files)}]", end=" ")
            try:
                result = self.predict(image_path, conf_threshold=conf_threshold)
                all_results.append(result)
            except Exception as e:
                print(f"ERROR processing {image_path.name}: {e}")
        
        # Overall summary
        print(f"\n{'='*60}")
        print(f"Batch Processing Complete")
        print(f"{'='*60}")
        print(f"Total images processed: {len(all_results)}/{len(image_files)}")
        
        return all_results
    
    def _draw_boxes(self, frame, detections):
        """
        Draw bounding boxes with confidence labels at corners and class label in top-left corner.
        
        Args:
            frame: OpenCV frame (numpy array)
            detections: List of detection dictionaries with bbox, class_name, confidence
            
        Returns:
            Annotated frame
        """
        if not detections:
            return frame
        
        # Define colors for each class (BGR format)
        colors = {
            'empty_drum': (0, 0, 255),      # Red
            'in_process': (0, 255, 0),      # Green
            'yellow_tire': (0, 255, 255)    # Yellow
        }
        
        # Display name mapping (internal name -> display name)
        display_names = {
            'empty_drum': 'EMPTY DRUM',
            'in_process': 'IN PROCESS',
            'yellow_tire': 'YELLOW COVER'
        }
        
        # Find detection with highest confidence
        highest_conf_det = max(detections, key=lambda d: d['confidence'])
        
        # Draw bounding boxes with confidence labels at corners
        for det in detections:
            bbox = det['bbox']
            class_name = det['class_name']
            confidence = det['confidence']
            
            # Get coordinates
            x1, y1, x2, y2 = map(int, bbox)
            
            # Get color for this class
            color = colors.get(class_name, (255, 255, 255))  # Default to white
            
            # Draw bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
            
            # Draw confidence percentage at top-left corner of bounding box
            conf_text = f"{confidence*100:.1f}%"
            font_scale = 0.7
            font_thickness = 2
            
            # Get text size for background
            (text_width, text_height), baseline = cv2.getTextSize(
                conf_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness
            )
            
            # Draw label background
            cv2.rectangle(
                frame,
                (x1, y1 - text_height - baseline - 5),
                (x1 + text_width + 5, y1),
                color,
                -1  # Filled
            )
            
            # Draw confidence text
            cv2.putText(
                frame,
                conf_text,
                (x1, y1 - baseline - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale,
                (0, 0, 0),  # Black text
                font_thickness,
                cv2.LINE_AA
            )
        
        # Draw only the highest confidence class label in top-left corner
        class_name = highest_conf_det['class_name']
        color = colors.get(class_name, (255, 255, 255))
        
        # Get display name (with yellow_tire -> Yellow Cover mapping)
        display_label = display_names.get(class_name, class_name.upper())
        
        # Get text size for background
        font_scale = 1.5
        font_thickness = 3
        (text_width, text_height), baseline = cv2.getTextSize(
            display_label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness
        )
        
        # Position in top-left corner with padding
        padding = 20
        x_pos = padding
        y_pos = padding + text_height
        
        # Draw label background (semi-transparent black background)
        overlay = frame.copy()
        cv2.rectangle(
            overlay,
            (x_pos - 10, y_pos - text_height - 10),
            (x_pos + text_width + 10, y_pos + baseline + 10),
            (0, 0, 0),
            -1
        )
        # Blend with original frame for transparency
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        
        # Draw label text in the color of the detected class
        cv2.putText(
            frame,
            display_label,
            (x_pos, y_pos),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            color,
            font_thickness,
            cv2.LINE_AA
        )
        
        return frame
    
    def predict_video(self, video_path, conf_threshold=0.25, sample_rate=1.0):
        """
        Run predictions on a video file and create annotated output.
        
        Args:
            video_path (str): Path to the video file
            conf_threshold (float): Confidence threshold for detections
            sample_rate (float): Process every N seconds (default: 1.0 = 1 fps)
            
        Returns:
            dict: Summary of video processing
        """
        video_path = Path(video_path)
        
        if not video_path.exists():
            raise FileNotFoundError(f"Video not found: {video_path}")
        
        print(f"\n{'='*60}")
        print(f"Analyzing Video: {video_path.name}")
        print(f"{'='*60}")
        print(f"Sample Rate: 1 frame every {sample_rate} second(s)")
        sys.stdout.flush()
        
        # Open video file
        cap = cv2.VideoCapture(str(video_path))
        
        if not cap.isOpened():
            raise RuntimeError(f"Failed to open video: {video_path}")
        
        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        duration = total_frames / fps if fps > 0 else 0
        
        print(f"Video Info:")
        print(f"  Resolution: {width}x{height}")
        print(f"  FPS: {fps:.2f}")
        print(f"  Total Frames: {total_frames}")
        print(f"  Duration: {duration:.1f} seconds ({duration/60:.1f} minutes)")
        sys.stdout.flush()
        
        # Calculate frames to process
        frame_skip = int(fps * sample_rate)
        estimated_frames = total_frames // frame_skip
        print(f"  Frames to process: ~{estimated_frames} (every {frame_skip} frames)")
        print(f"\nProcessing... This may take several minutes.")
        sys.stdout.flush()
        
        # Setup output video
        output_dir = Path("results/predictions")
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"{video_path.stem}_annotated{video_path.suffix}"
        
        # Use same codec as input or fallback to mp4v
        # Save at 8x speed by multiplying FPS by 8
        output_fps = fps * 8
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(str(output_path), fourcc, output_fps, (width, height))
        
        print(f"  Output FPS: {output_fps:.2f} (8x speed)")
        sys.stdout.flush()
        
        # Process video
        frame_num = 0
        processed_count = 0
        detection_stats = {
            'empty_drum': 0,
            'in_process': 0,
            'yellow_tire': 0
        }
        
        # Store last detection results to reuse on subsequent frames
        last_detections = None
        
        try:
            while True:
                ret, frame = cap.read()
                
                if not ret:
                    break
                
                # Process frame at sample rate
                if frame_num % frame_skip == 0:
                    # Run inference on this frame
                    try:
                        # Save frame temporarily
                        temp_frame_path = output_dir / f"temp_frame_{frame_num}.jpg"
                        cv2.imwrite(str(temp_frame_path), frame)
                        
                        # Run prediction
                        results = self.model(
                            source=str(temp_frame_path),
                            conf=conf_threshold,
                            verbose=False,
                            device='cpu',
                            stream=False
                        )
                        
                        result = results[0]
                        
                        # Count detections
                        for box in result.boxes:
                            class_id = int(box.cls[0])
                            class_name = self.class_names[class_id]
                            detection_stats[class_name] += 1
                        
                        # Store detection information for reuse
                        last_detections = []
                        for box in result.boxes:
                            class_id = int(box.cls[0])
                            confidence = float(box.conf[0])
                            bbox = box.xyxy[0].tolist()  # [x1, y1, x2, y2]
                            class_name = self.class_names[class_id]
                            
                            last_detections.append({
                                'class_id': class_id,
                                'class_name': class_name,
                                'confidence': confidence,
                                'bbox': bbox
                            })
                        
                        # Clean up temp file (non-blocking on Windows file locks)
                        try:
                            temp_frame_path.unlink()
                        except (PermissionError, OSError) as e:
                            # File might be locked by another process on Windows - ignore and continue
                            pass
                        
                        processed_count += 1
                        
                        # Progress update
                        if processed_count % 10 == 0:
                            progress = (frame_num / total_frames) * 100
                            print(f"  Progress: {progress:.1f}% ({processed_count} frames processed)", flush=True)
                        
                    except Exception as e:
                        print(f"  Warning: Error processing frame {frame_num}: {e}", flush=True)
                        # Keep last detections even on error
                
                # Draw bounding boxes using last detection results
                if last_detections is not None:
                    annotated_frame = self._draw_boxes(frame.copy(), last_detections)
                else:
                    annotated_frame = frame.copy()
                
                # Write annotated frame
                out.write(annotated_frame)
                
                frame_num += 1
            
        finally:
            cap.release()
            out.release()
        
        # Summary
        summary = {
            'video_path': str(video_path),
            'output_path': str(output_path),
            'total_frames': total_frames,
            'processed_frames': processed_count,
            'duration': duration,
            'detection_stats': detection_stats
        }
        
        print(f"\n{'='*60}")
        print(f"Video Processing Complete!")
        print(f"{'='*60}")
        print(f"Processed: {processed_count} frames")
        print(f"\nDetection Statistics:")
        for class_name, count in detection_stats.items():
            print(f"  - {class_name}: {count} detections")
        print(f"\n✓ Annotated video saved to: {output_path}")
        sys.stdout.flush()
        
        return summary


def main():
    """Main entry point for the tire drum classification system."""
    
    print("="*60)
    print("Tire Drum Classification System")
    print("="*60)
    print("Classes: empty_drum, in_process, yellow_tire")
    print("="*60)
    
    # Configuration
    MODEL_PATH = "YOLO/best.pt"
    CONFIDENCE_THRESHOLD = 0.25  # Adjust this to filter low-confidence detections
    
    # Initialize classifier
    try:
        classifier = TireDrumClassifier(model_path=MODEL_PATH)
    except Exception as e:
        print(f"\n✗ Error initializing classifier: {e}")
        sys.exit(1)
    
    # Interactive loop
    while True:
        print(f"\n{'='*60}")
        print("Options:")
        print("  1. Analyze a single image")
        print("  2. Analyze all images in a folder")
        print("  3. Analyze a video")
        print("  4. Exit")
        print("="*60)
        
        choice = input("Enter your choice (1/2/3/4): ").strip()
        
        if choice == "1":
            # Single image analysis
            image_path = input("\nEnter the path to the image: ").strip()
            
            # Remove quotes if user copied path with quotes
            image_path = image_path.strip('"').strip("'")
            
            if not image_path:
                print("✗ No path provided. Please try again.")
                continue
            
            try:
                classifier.predict(image_path, conf_threshold=CONFIDENCE_THRESHOLD)
            except Exception as e:
                print(f"\n✗ Error: {e}")
        
        elif choice == "2":
            # Batch folder analysis
            folder_path = input("\nEnter the path to the folder: ").strip()
            
            # Remove quotes if user copied path with quotes
            folder_path = folder_path.strip('"').strip("'")
            
            if not folder_path:
                print("✗ No path provided. Please try again.")
                continue
            
            try:
                classifier.batch_predict(folder_path, conf_threshold=CONFIDENCE_THRESHOLD)
            except Exception as e:
                print(f"\n✗ Error: {e}")
        
        elif choice == "3":
            # Video analysis
            video_path = input("\nEnter the path to the video: ").strip()
            
            # Remove quotes if user copied path with quotes
            video_path = video_path.strip('"').strip("'")
            
            if not video_path:
                print("✗ No path provided. Please try again.")
                continue
            
            try:
                # Ask for sample rate (default 1 fps)
                sample_input = input("Sample rate in seconds (press Enter for 1.0 fps): ").strip()
                sample_rate = float(sample_input) if sample_input else 1.0
                
                classifier.predict_video(video_path, conf_threshold=CONFIDENCE_THRESHOLD, sample_rate=sample_rate)
            except ValueError:
                print("\n✗ Invalid sample rate. Using default 1.0 second.")
                classifier.predict_video(video_path, conf_threshold=CONFIDENCE_THRESHOLD, sample_rate=1.0)
            except Exception as e:
                print(f"\n✗ Error: {e}")
        
        elif choice == "4":
            print("\nExiting. Goodbye!")
            break
        
        else:
            print("\n✗ Invalid choice. Please enter 1, 2, 3, or 4.")


if __name__ == "__main__":
    main()

