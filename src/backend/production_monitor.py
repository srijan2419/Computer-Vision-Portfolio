"""
Backend Integration Script
Connects the RTSP stream capture and YOLO detection with the web application database
"""

import os
import sys
import time
import json
from datetime import datetime
from pathlib import Path
from threading import Thread
import cv2

# Add parent directory to path to import main modules
sys.path.insert(0, str(Path(__file__).parent.parent))

from app import app, db, TireProduction, ProductionLog, SystemStatus, SystemSettings, Shift, get_next_daily_number

# Force CPU mode
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

import torch
from ultralytics import YOLO

torch.cuda.is_available = lambda: False


class ProductionMonitor:
    """Monitors RTSP stream and logs tire production to database"""
    
    def __init__(self, rtsp_url, model_path="../YOLO/best.pt"):
        self.rtsp_url = rtsp_url
        self.model_path = Path(model_path)
        self.model = None
        self.running = False
        self.cap = None
        
        # Production tracking
        self.current_tire_id = None
        self.current_state = None
        self.state_start_time = None
        self.frame_buffer = []  # Store frames for tire photos
        
        # Load YOLO model
        self._load_model()
        
    def _load_model(self):
        """Load YOLO model"""
        print(f"Loading model from: {self.model_path}")
        self.model = YOLO(str(self.model_path))
        print("✓ Model loaded successfully!")
        
    def _connect_to_stream(self):
        """Connect to RTSP stream - returns True on success, False on failure (never raises)"""
        print(f"Connecting to RTSP stream: {self.rtsp_url}")
        
        try:
            if self.cap is not None:
                self.cap.release()
            
            # Use CAP_FFMPEG like in stream_capture.py
            self.cap = cv2.VideoCapture(self.rtsp_url, cv2.CAP_FFMPEG)
            
            # Set buffer size to minimize latency
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            
            if not self.cap.isOpened():
                print(f"❌ Failed to open RTSP stream")
                return False
            
            # Try to read one frame to verify stream is working
            ret, frame = self.cap.read()
            if not ret or frame is None:
                print(f"❌ Stream opened but cannot read frames")
                return False
            
            # Get stream properties
            fps = self.cap.get(cv2.CAP_PROP_FPS)
            width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            print(f"✓ Connected - Resolution: {width}x{height}, FPS: {fps}")
            return True
            
        except Exception as e:
            print(f"❌ Connection exception: {e}")
            return False
        
    def _save_frame_image(self, frame, tire_id):
        """Save frame to disk and return filename"""
        output_dir = Path("../results/predictions")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
        filename = f"tire_{tire_id}_{timestamp}.jpg"
        filepath = output_dir / filename
        
        cv2.imwrite(str(filepath), frame)
        return filename
        
    def _get_current_shift_id(self):
        """Get current shift ID based on time (only active shifts)"""
        now_time = datetime.now().time()
        shifts = Shift.query.filter_by(is_deleted=False).all()
        
        for shift in shifts:
            # Check if shift spans across midnight
            if shift.start_time <= shift.end_time:
                if shift.start_time <= now_time <= shift.end_time:
                    return shift.id
            else:
                # Shift spans midnight (e.g. 22:00 to 06:00)
                if now_time >= shift.start_time or now_time <= shift.end_time:
                    return shift.id
        return None

    def _start_new_tire(self):
        """Start tracking a new tire production"""
        with app.app_context():
            # CRITICAL: Check if another tire is already in process
            existing = TireProduction.query.filter_by(status='in_process').first()
            if existing:
                print(f"⚠️  WARNING: Tire #{existing.id} already in process! Not creating new tire.")
                print(f"    Will resume tracking existing tire instead.")
                self.current_tire_id = existing.id
                self.state_start_time = existing.start_time
                self.frame_buffer = []  # Fresh buffer
                return existing.id
            
            # Get current shift and daily number
            shift_id = self._get_current_shift_id()
            daily_num = get_next_daily_number()
            
            tire = TireProduction(
                daily_number=daily_num,
                start_time=datetime.now(),
                status='in_process',
                shift_id=shift_id
            )
            db.session.add(tire)
            db.session.commit()
            
            self.current_tire_id = tire.id
            self.state_start_time = datetime.now()
            self.frame_buffer = []
            
            print(f"✓ Started new tire production: Tire #{tire.id} (Daily #{daily_num}, Shift ID: {shift_id})")
            return tire.id
            
    def _complete_tire(self):
        """Complete current tire production"""
        if not self.current_tire_id:
            return
        
        with app.app_context():
            tire = TireProduction.query.get(self.current_tire_id)
            if tire:
                tire.end_time = datetime.now()
                tire.duration_seconds = (tire.end_time - tire.start_time).seconds
                
                # Check minimum duration threshold
                settings = SystemSettings.query.first()
                minimum_duration_seconds = (settings.minimum_duration_minutes * 60) if settings else 300  # Default 5 minutes
                
                if tire.duration_seconds < minimum_duration_seconds:
                    # Duration too short - delete this tire and its logs
                    print(f"⚠️  FILTERED OUT Tire #{tire.id}: Duration {tire.duration_seconds}s ({tire.duration_seconds/60:.1f}min) < minimum {minimum_duration_seconds}s ({minimum_duration_seconds/60}min)")
                    
                    # Delete associated production logs
                    ProductionLog.query.filter_by(tire_id=tire.id).delete()
                    
                    # Delete the tire itself
                    db.session.delete(tire)
                    db.session.commit()
                else:
                    # Duration acceptable - complete normally
                    tire.status = 'completed'
                    
                    # Calculate average confidence
                    logs = ProductionLog.query.filter_by(tire_id=tire.id).all()
                    if logs:
                        tire.confidence_avg = sum([log.confidence for log in logs]) / len(logs)
                    
                    # Save photos using time-based selection strategy
                    if self.frame_buffer:
                        photo_filenames = []
                        buffer_size = len(self.frame_buffer)
                        duration_seconds = tire.duration_seconds
                        
                        if buffer_size == 0:
                            print(f"   ⚠️  No frames in buffer to save")
                        elif buffer_size <= 3:
                            # Very short production - save all frames
                            selected_frames = self.frame_buffer
                            for frame in selected_frames:
                                filename = self._save_frame_image(frame, tire.id)
                                photo_filenames.append(filename)
                            print(f"   Saved {len(photo_filenames)} photos (all frames - short production)")
                        else:
                            # Time-based frame selection strategy
                            # Each frame represents a specific timestamp in the production timeline
                            
                            # Calculate time per frame (in seconds)
                            time_per_frame = duration_seconds / buffer_size
                            
                            # Dictionary to store {timestamp: frame_index}
                            selected_indices = {}
                            
                            # 1. FIRST FRAME - Always include (timestamp = 0)
                            selected_indices[0.0] = 0
                            
                            # 2. MIDDLE FRAMES - At 60-second intervals
                            current_interval = 60  # Start at 60 seconds
                            while current_interval < duration_seconds - 30:  # Stop before we reach last 30s
                                # Find the frame closest to this timestamp
                                target_frame_index = int(current_interval / time_per_frame)
                                # Ensure index is within bounds (not first, not last)
                                if 0 < target_frame_index < buffer_size - 1:
                                    actual_timestamp = target_frame_index * time_per_frame
                                    selected_indices[actual_timestamp] = target_frame_index
                                current_interval += 60
                            
                            # 3. PENULTIMATE FRAME - 30 seconds before the last frame
                            penultimate_timestamp = duration_seconds - 30
                            if penultimate_timestamp > 0:
                                # Find the frame CLOSEST to 30s before end (use rounding instead of int)
                                penultimate_frame_index = round(penultimate_timestamp / time_per_frame)
                                # Ensure it's a valid index and not the same as last frame
                                if 0 < penultimate_frame_index < buffer_size - 1:
                                    # Check if we already have a frame within 30 seconds of the last frame
                                    # Get all currently selected timestamps
                                    existing_timestamps = sorted(selected_indices.keys())
                                    
                                    # Check if any existing timestamp (except the first) is within 30s of the end
                                    add_penultimate = True
                                    for ts in existing_timestamps:
                                        if ts > 0:  # Skip the first frame timestamp
                                            time_from_end = duration_seconds - ts
                                            if time_from_end <= 30 and time_from_end > 0:
                                                # Already have a frame within 30s of end
                                                add_penultimate = False
                                                break
                                    
                                    if add_penultimate:
                                        actual_penultimate_timestamp = penultimate_frame_index * time_per_frame
                                        selected_indices[actual_penultimate_timestamp] = penultimate_frame_index
                            
                            # 4. LAST FRAME - Always include
                            last_timestamp = duration_seconds
                            selected_indices[last_timestamp] = buffer_size - 1
                            
                            # Sort by timestamp and extract frames
                            sorted_timestamps = sorted(selected_indices.keys())
                            selected_frames_with_info = []
                            
                            for timestamp in sorted_timestamps:
                                frame_index = selected_indices[timestamp]
                                frame = self.frame_buffer[frame_index]
                                selected_frames_with_info.append({
                                    'frame': frame,
                                    'index': frame_index,
                                    'timestamp': timestamp
                                })
                            
                            # Save all selected frames
                            for info in selected_frames_with_info:
                                filename = self._save_frame_image(info['frame'], tire.id)
                                photo_filenames.append(filename)
                            
                            # Debug output
                            print(f"   Saved {len(photo_filenames)} photos from {buffer_size} total frames (duration: {duration_seconds}s)")
                            timestamps = [f"{info['timestamp']:.1f}s" for info in selected_frames_with_info]
                            print(f"   Frame selection timestamps: {timestamps}")
                        
                        tire.photos = json.dumps(photo_filenames)
                    
                    db.session.commit()
                    
                    print(f"✓ Completed tire production: Tire #{tire.id}, Duration: {tire.duration_seconds}s ({tire.duration_seconds/60:.1f}min)")
        
        self.current_tire_id = None
        self.frame_buffer = []
    
    def _resume_incomplete_tire(self):
        """Resume tracking of existing incomplete tire from database"""
        with app.app_context():
            incomplete_tire = TireProduction.query.filter_by(status='in_process').first()
            if incomplete_tire:
                self.current_tire_id = incomplete_tire.id
                self.state_start_time = incomplete_tire.start_time
                self.frame_buffer = []  # Fresh buffer (old frames lost, but better than phantom)
                self.current_state = 'in_process'  # Assume we're mid-production
                print(f"⚠️  RESUMED incomplete tire from database: Tire #{incomplete_tire.id}")
                print(f"    Started at: {incomplete_tire.start_time}")
                return True
        return False
        
    def _log_detection(self, detection_class, confidence, frame_path=None):
        """Log detection to database"""
        with app.app_context():
            log = ProductionLog(
                detection_class=detection_class,
                confidence=confidence,
                frame_path=frame_path,
                tire_id=self.current_tire_id
            )
            db.session.add(log)
            db.session.commit()
            
    def _save_live_frame(self, frame):
        """Save current frame for live feed display"""
        output_dir = Path("../results/predictions")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Always save to same filename for live feed
        filename = "live_feed_current.jpg"
        filepath = output_dir / filename
        
        cv2.imwrite(str(filepath), frame)
        return filename
            
    def _update_system_status(self, detection_class, frames_processed):
        """Update system status in database"""
        with app.app_context():
            status = SystemStatus.query.first()
            if not status:
                status = SystemStatus()
                db.session.add(status)
            
            status.current_detection = detection_class
            status.last_update = datetime.now()
            status.frames_processed = frames_processed
            status.stream_active = self.running
            
            db.session.commit()
            
    def _process_detection(self, detection_class, confidence, frame):
        """
        Process detection and update state machine.
        Correct sequence: RED (empty_drum) → YELLOW (yellow_tire) → GREEN (in_process) → YELLOW → RED
        One complete cycle = One tire produced
        """
        
        prev_state = self.current_state
        prev_tire_id = self.current_tire_id
        
        # State machine logic with correct phase sequence
        if detection_class == 'empty_drum':  # RED PHASE
            # Complete tire if coming back from yellow (final yellow phase)
            if self.current_tire_id and self.current_state == 'yellow_tire':
                print(f"🔴 State transition: YELLOW → RED (Completing Tire #{self.current_tire_id})")
                self._complete_tire()
            elif self.current_tire_id:
                # Empty drum seen but tire still active - complete it regardless of previous state
                print(f"🔴 RED phase detected - Completing Tire #{self.current_tire_id} (was in {self.current_state} state)")
                self._complete_tire()
            
            self.current_state = 'empty_drum'
            
        elif detection_class == 'yellow_tire':  # YELLOW PHASE (middle phase)
            # If coming from red (empty), this is first yellow - start new tire
            if self.current_state == 'empty_drum' and not self.current_tire_id:
                print(f"🟡 State transition: RED → YELLOW (Starting new tire)")
                self._start_new_tire()
            elif not self.current_tire_id:
                print(f"🟡 YELLOW detected without active tire (prev state: {prev_state}) - will start tire")
                self._start_new_tire()
            
            # Store frames during yellow phase (limit to 100 frames)
            if self.current_tire_id and len(self.frame_buffer) < 100:
                self.frame_buffer.append(frame.copy())
            
            self.current_state = 'yellow_tire'
            
        elif detection_class == 'in_process':  # GREEN PHASE
            # Start new tire if somehow we skipped yellow (error recovery)
            if not self.current_tire_id:
                print(f"🟢 GREEN detected without active tire (prev state: {prev_state}) - ERROR RECOVERY")
                self._start_new_tire()
            
            # Store frames while in green process phase (limit to 100 frames)
            if self.current_tire_id and len(self.frame_buffer) < 100:
                self.frame_buffer.append(frame.copy())
            
            self.current_state = 'in_process'
        
        # Log state transition if it changed
        if prev_state != self.current_state or prev_tire_id != self.current_tire_id:
            print(f"   State: {prev_state or 'None'} → {self.current_state} | Tire: {prev_tire_id or 'None'} → {self.current_tire_id or 'None'}")
        
        # NOTE: Detection logging removed from here to prevent duplicates
        # Logging is handled in the main loop (line ~287)
            
    def start(self):
        """Start monitoring the RTSP stream"""
        self.running = True
        
        print("="*60)
        print("Starting Production Monitor")
        print("="*60)
        
        # Check for incomplete tires and resume if found
        if self._resume_incomplete_tire():
            print("    System will continue tracking the resumed tire")
        else:
            print("    No incomplete tires found, starting fresh")
        
        # Connect to stream
        if not self._connect_to_stream():
            print("Failed to connect to stream")
            return
        
        frame_count = 0
        last_process_time = 0
        process_interval = 1.0  # Process 1 frame per second
        
        try:
            while self.running:
                ret, frame = self.cap.read()
                
                if not ret:
                    print("Failed to read frame, attempting reconnection...")
                    time.sleep(5)
                    # Keep retrying indefinitely - don't break the loop
                    reconnect_attempt = 1
                    while self.running and not self._connect_to_stream():
                        print(f"Reconnection attempt {reconnect_attempt} failed. Retrying in 10 seconds...")
                        time.sleep(10)
                        reconnect_attempt += 1
                    
                    if self.running:
                        print(f"✓ Reconnected successfully after {reconnect_attempt} attempts")
                    continue
                
                current_time = time.time()
                
                # Process frame at interval
                if current_time - last_process_time >= process_interval:
                    frame_count += 1
                    temp_path = None
                    
                    try:
                        # Save frame temporarily for prediction
                        temp_path = Path("temp_frame.jpg")
                        cv2.imwrite(str(temp_path), frame)
                        
                        # Run YOLO detection
                        results = self.model(str(temp_path), device='cpu', verbose=False, conf=0.25)
                        result = results[0]
                        
                        # Get highest confidence detection
                        if len(result.boxes) > 0:
                            confidences = [float(box.conf[0]) for box in result.boxes]
                            class_ids = [int(box.cls[0]) for box in result.boxes]
                            
                            max_idx = confidences.index(max(confidences))
                            detection_class = ['empty_drum', 'in_process', 'yellow_tire'][class_ids[max_idx]]
                            confidence = confidences[max_idx]
                            
                            # Save current frame for live feed
                            live_frame_filename = self._save_live_frame(frame)
                            
                            # Process the detection
                            self._process_detection(detection_class, confidence, frame)
                            
                            # Log with frame path for live feed
                            if self.current_tire_id:
                                self._log_detection(detection_class, confidence, live_frame_filename)
                            
                            # Update system status
                            self._update_system_status(detection_class, frame_count)
                            
                            # Display phase color in console
                            phase_color = "🔴 RED" if detection_class == 'empty_drum' else "🟡 YELLOW" if detection_class == 'yellow_tire' else "🟢 GREEN"
                            print(f"Frame {frame_count}: {phase_color} - {detection_class} ({confidence:.2%})")
                    
                    except Exception as e:
                        # Non-fatal errors in frame processing - log and continue
                        print(f"⚠️  Warning: Error processing frame {frame_count}: {e}")
                        # Continue to next frame instead of crashing
                    
                    finally:
                        # Clean up temp file (non-blocking on Windows file locks)
                        if temp_path and temp_path.exists():
                            try:
                                temp_path.unlink()
                            except (PermissionError, OSError):
                                # File might be locked by another process on Windows - ignore and continue
                                pass
                        
                        last_process_time = current_time
                
                # Small sleep to prevent CPU spinning
                time.sleep(0.01)
                
        except KeyboardInterrupt:
            print("\nKeyboard interrupt received. Shutting down...")
        except Exception as e:
            print(f"Error in monitoring loop: {e}")
            import traceback
            traceback.print_exc()
        finally:
            self.stop()
            
    def stop(self):
        """Stop monitoring"""
        print("Stopping production monitor...")
        self.running = False
        
        # Complete any active tire
        if self.current_tire_id:
            self._complete_tire()
        
        if self.cap:
            self.cap.release()
        
        # Update system status
        with app.app_context():
            status = SystemStatus.query.first()
            if status:
                status.stream_active = False
                status.last_update = datetime.now()
                db.session.commit()
        
        print("✓ Production monitor stopped")
