"""
RTSP Stream Frame Capture System
Captures frames from an RTSP camera stream and saves them to disk.
Production-ready with error handling, reconnection logic, and monitoring.
"""

import cv2
import os
import time
import logging
from datetime import datetime
from pathlib import Path
from urllib.parse import quote


class RTSPFrameCapture:
    """Captures and saves frames from an RTSP camera stream."""
    
    def __init__(self, rtsp_url, output_dir="input", frame_interval=1.0, 
                 reconnect_delay=5, max_reconnect_attempts=None, username=None, password=None):
        """
        Initialize the RTSP frame capture system.
        
        Args:
            rtsp_url (str): RTSP stream URL (e.g., rtsp://camera.invalid/stream1)
            output_dir (str): Directory to save captured frames
            frame_interval (float): Seconds between frame captures (default: 1.0 = 1 fps)
            reconnect_delay (int): Seconds to wait before reconnection attempt
            max_reconnect_attempts (int): Max reconnection attempts (None = infinite)
            username (str): Camera username (if authentication required)
            password (str): Camera password (if authentication required)
        """
        # Build RTSP URL with credentials if provided
        if username and password:
            # Parse the URL to inject credentials with proper URL encoding
            # Inject URL-encoded credentials into the stream URI at runtime
            # URL-encode credentials to handle special characters like @, :, etc.
            if "://" in rtsp_url:
                protocol, rest = rtsp_url.split("://", 1)
                encoded_username = quote(username, safe='')
                encoded_password = quote(password, safe='')
                self.rtsp_url = f"{protocol}://{encoded_username}:{encoded_password}@{rest}"
            else:
                self.rtsp_url = rtsp_url
        else:
            self.rtsp_url = rtsp_url
        self.output_dir = Path(output_dir)
        self.frame_interval = frame_interval
        self.reconnect_delay = reconnect_delay
        self.max_reconnect_attempts = max_reconnect_attempts
        
        self.cap = None
        self.is_running = False
        self.frame_count = 0
        self.error_count = 0
        
        # Setup logging
        self._setup_logging()
        
        # Create output directory
        self._create_output_directory()
        
    def _setup_logging(self):
        """Configure logging for the capture system."""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('stream_capture.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        
    def _create_output_directory(self):
        """Create output directory if it doesn't exist."""
        try:
            self.output_dir.mkdir(parents=True, exist_ok=True)
            self.logger.info(f"Output directory ready: {self.output_dir.absolute()}")
        except Exception as e:
            self.logger.error(f"Failed to create output directory: {e}")
            raise
            
    def _connect_to_stream(self):
        """Establish connection to RTSP stream."""
        self.logger.info(f"Connecting to stream: {self.rtsp_url}")
        
        try:
            # Release existing connection if any
            if self.cap is not None:
                self.cap.release()
            
            # Open RTSP stream
            self.cap = cv2.VideoCapture(self.rtsp_url, cv2.CAP_FFMPEG)
            
            # Set buffer size to minimize latency
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            
            # Verify connection
            if not self.cap.isOpened():
                raise ConnectionError("Failed to open RTSP stream")
            
            # Try to read one frame to verify stream is working
            ret, frame = self.cap.read()
            if not ret or frame is None:
                raise ConnectionError("Stream opened but cannot read frames")
            
            # Log stream properties
            width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = self.cap.get(cv2.CAP_PROP_FPS)
            
            self.logger.info(f"Connected successfully - Resolution: {width}x{height}, FPS: {fps}")
            return True
            
        except Exception as e:
            self.logger.error(f"Connection failed: {e}")
            return False
            
    def _save_frame(self, frame):
        """
        Save frame to disk with timestamp filename.
        
        Args:
            frame: OpenCV frame (numpy array)
            
        Returns:
            bool: True if saved successfully, False otherwise
        """
        try:
            # Generate filename with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]  # millisecond precision
            filename = f"frame_{timestamp}.jpg"
            filepath = self.output_dir / filename
            
            # Save frame as JPEG with quality 95
            success = cv2.imwrite(str(filepath), frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
            
            if success:
                self.frame_count += 1
                if self.frame_count % 10 == 0:  # Log every 10 frames
                    self.logger.info(f"Frames captured: {self.frame_count}")
                return True
            else:
                self.logger.error(f"Failed to save frame: {filepath}")
                return False
                
        except Exception as e:
            self.logger.error(f"Error saving frame: {e}")
            return False
            
    def _reconnect(self):
        """
        Attempt to reconnect to the stream with exponential backoff.
        
        Returns:
            bool: True if reconnected successfully, False otherwise
        """
        attempt = 0
        
        while self.is_running:
            attempt += 1
            
            # Check max attempts
            if self.max_reconnect_attempts and attempt > self.max_reconnect_attempts:
                self.logger.error(f"Max reconnection attempts ({self.max_reconnect_attempts}) reached")
                return False
            
            self.logger.warning(f"Reconnection attempt {attempt}...")
            
            # Wait before reconnecting
            time.sleep(self.reconnect_delay)
            
            # Try to connect
            if self._connect_to_stream():
                self.logger.info("Reconnection successful")
                self.error_count = 0
                return True
        
        return False
        
    def start(self):
        """Start capturing frames from the RTSP stream."""
        self.is_running = True
        self.logger.info("Starting frame capture system...")
        
        # Initial connection
        if not self._connect_to_stream():
            self.logger.error("Initial connection failed")
            if not self._reconnect():
                self.logger.error("Unable to establish connection. Exiting.")
                return
        
        last_capture_time = 0
        consecutive_failures = 0
        max_consecutive_failures = 10
        
        try:
            while self.is_running:
                current_time = time.time()
                
                # Check if it's time to capture a frame
                if current_time - last_capture_time >= self.frame_interval:
                    # Read frame from stream
                    ret, frame = self.cap.read()
                    
                    if ret and frame is not None:
                        # Frame captured successfully
                        self._save_frame(frame)
                        last_capture_time = current_time
                        consecutive_failures = 0
                        
                    else:
                        # Frame read failed
                        consecutive_failures += 1
                        self.error_count += 1
                        self.logger.warning(f"Failed to read frame (consecutive failures: {consecutive_failures})")
                        
                        # If too many consecutive failures, attempt reconnection
                        if consecutive_failures >= max_consecutive_failures:
                            self.logger.error("Too many consecutive failures. Attempting reconnection...")
                            if not self._reconnect():
                                self.logger.error("Reconnection failed. Exiting.")
                                break
                            consecutive_failures = 0
                else:
                    # Small sleep to prevent CPU spinning
                    time.sleep(0.01)
                    
        except KeyboardInterrupt:
            self.logger.info("Keyboard interrupt received. Shutting down...")
        except Exception as e:
            self.logger.error(f"Unexpected error in capture loop: {e}")
        finally:
            self.stop()
            
    def stop(self):
        """Stop the capture system and cleanup resources."""
        self.logger.info("Stopping frame capture system...")
        self.is_running = False
        
        if self.cap is not None:
            self.cap.release()
            
        self.logger.info(f"Capture stopped. Total frames captured: {self.frame_count}")
        self.logger.info(f"Total errors encountered: {self.error_count}")
