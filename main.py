import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
import tempfile
import os
import pytesseract
import time

class LaneDetector:
    def __init__(self):
        self.prev_lines = None
        
    def detect_lanes(self, frame):
        try:
            # Convert frame to grayscale
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Apply Gaussian blur
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            
            # Edge detection
            edges = cv2.Canny(blurred, 50, 150)
            
            # Define region of interest (ROI)
            height, width = frame.shape[:2]
            roi_vertices = np.array([
                [(0, height), (width/2, height*0.6), (width, height)]
            ], dtype=np.int32)
            
            # Create mask for ROI
            mask = np.zeros_like(edges)
            cv2.fillPoly(mask, roi_vertices, 255)
            masked_edges = cv2.bitwise_and(edges, mask)
            
            # Detect lines using Hough transform
            lines = cv2.HoughLinesP(
                masked_edges,
                rho=1,
                theta=np.pi/180,
                threshold=20,
                minLineLength=50,
                maxLineGap=50
            )
            
            if lines is not None:
                self.prev_lines = lines
            elif self.prev_lines is not None:
                lines = self.prev_lines
                
            return lines
        except Exception as e:
            print(f"Error in lane detection: {e}")
            return None

    def calculate_direction(self, lines, frame_width):
        if lines is None:
            return "forward"
            
        left_lines = []
        right_lines = []
        mid_x = frame_width // 2
        
        for line in lines:
            x1, y1, x2, y2 = line[0]
            slope = (y2 - y1) / (x2 - x1) if x2 != x1 else 0
            
            if slope < 0:  # Left lane
                left_lines.append((x1 + x2) / 2)
            elif slope > 0:  # Right lane
                right_lines.append((x1 + x2) / 2)
        
        if len(left_lines) > 0 and len(right_lines) > 0:
            avg_left = sum(left_lines) / len(left_lines)
            avg_right = sum(right_lines) / len(right_lines)
            center = (avg_left + avg_right) / 2
            
            if center < mid_x - 50:
                return "left"
            elif center > mid_x + 50:
                return "right"
        
        return "forward"

    def draw_direction_arrow(self, frame, direction):
        height, width = frame.shape[:2]
        center_x = width // 2
        center_y = height - 100
        arrow_length = 100
        color = (0, 255, 255)  # Yellow color
        thickness = 3
        
        if direction == "forward":
            pt1 = (center_x, center_y + arrow_length)
            pt2 = (center_x, center_y)
            pt3 = (center_x - 20, center_y + 20)
            pt4 = (center_x + 20, center_y + 20)
            cv2.arrowedLine(frame, pt1, pt2, color, thickness)
            cv2.putText(frame, "GO FORWARD", (center_x - 60, center_y - 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
            
        elif direction == "left":
            pt1 = (center_x + arrow_length//2, center_y)
            pt2 = (center_x - arrow_length//2, center_y)
            cv2.arrowedLine(frame, pt1, pt2, color, thickness)
            cv2.putText(frame, "TURN LEFT", (center_x - 50, center_y - 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
            
        elif direction == "right":
            pt1 = (center_x - arrow_length//2, center_y)
            pt2 = (center_x + arrow_length//2, center_y)
            cv2.arrowedLine(frame, pt1, pt2, color, thickness)
            cv2.putText(frame, "TURN RIGHT", (center_x - 50, center_y - 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

class VehicleDetector:
    def __init__(self):
        self.model = YOLO("yolov8n.pt")
        self.confidence_threshold = 0.5
        self.vehicle_classes = {2: 'Car', 3: 'Motorcycle', 5: 'Bus', 7: 'Truck'}
        self.traffic_signals = {9: 'Traffic Light'}
        self.current_status = "GO"
        self.lane_detector = LaneDetector()
        # Add two proximity thresholds
        self.stop_threshold = 0.4  # Very close - trigger stop
        self.slow_threshold = 0.25  # Moderately close - trigger slow

    def detect_license_plate(self, plate_img):
        if plate_img is None or plate_img.size == 0:
            return None
        try:
            gray = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)
            thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                         cv2.THRESH_BINARY_INV, 11, 2)
            text = pytesseract.image_to_string(thresh, config='--psm 6')
            return text.strip() if text else None
        except Exception as e:
            print(f"Error reading plate: {e}")
            return None

    def detect_traffic_light_color(self, frame, box):
        try:
            x1, y1, x2, y2 = map(int, box)
            light_region = frame[y1:y2, x1:x2]
            hsv = cv2.cvtColor(light_region, cv2.COLOR_BGR2HSV)
            
            red_mask = cv2.inRange(hsv, np.array([0, 100, 100]), np.array([10, 255, 255]))
            green_mask = cv2.inRange(hsv, np.array([40, 100, 100]), np.array([90, 255, 255]))
            
            red_pixels = np.sum(red_mask > 0)
            green_pixels = np.sum(green_mask > 0)
            
            if red_pixels > green_pixels and red_pixels > 100:
                self.current_status = "STOP"
                return "red"
            elif green_pixels > red_pixels and green_pixels > 100:
                self.current_status = "GO"
                return "green"
            return "unknown"
        except:
            return "unknown"

    def check_proximity(self, box, frame_height):
        """Check if vehicle is too close based on its bounding box size"""
        _, y1, _, y2 = map(int, box)
        vehicle_height = y2 - y1
        height_ratio = vehicle_height / frame_height
        
        if height_ratio > self.stop_threshold:
            return "STOP"
        elif height_ratio > self.slow_threshold:
            return "SLOW"
        return "GO"

    def process_frame(self, frame):
        # Detect lanes
        lanes = self.lane_detector.detect_lanes(frame)
        
        # Calculate direction based on lanes
        direction = self.lane_detector.calculate_direction(lanes, frame.shape[1])
        
        # Draw lanes
        if lanes is not None:
            for line in lanes:
                x1, y1, x2, y2 = line[0]
                cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        # Draw direction arrow
        self.lane_detector.draw_direction_arrow(frame, direction)
        
        # Object detection
        results = self.model(frame, conf=self.confidence_threshold)
        self.current_status = "GO"  # Reset status
        proximity_status = "GO"
        
        frame_height = frame.shape[0]
        
        for r in results:
            boxes = r.boxes
            for box in boxes:
                cls = int(box.cls[0])
                conf = float(box.conf[0])
                
                if cls in self.vehicle_classes or cls in self.traffic_signals:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    
                    if cls in self.vehicle_classes:
                        # Check proximity for vehicles
                        vehicle_proximity = self.check_proximity(box.xyxy[0], frame_height)
                        
                        if vehicle_proximity == "STOP":
                            proximity_status = "STOP"
                            self.current_status = "STOP - VEHICLE TOO CLOSE"
                            # Draw red box for close vehicles
                            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 3)
                            # Add warning text
                            cv2.putText(frame, "WARNING: VEHICLE TOO CLOSE!", 
                                      (frame.shape[1]//2 - 200, frame.shape[0]//2),
                                      cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
                        elif vehicle_proximity == "SLOW" and proximity_status != "STOP":
                            proximity_status = "SLOW"
                            self.current_status = "SLOW DOWN - VEHICLE AHEAD"
                            # Draw yellow box for vehicles at moderate distance
                            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)
                            # Add warning text
                            cv2.putText(frame, "CAUTION: SLOW DOWN!", 
                                      (frame.shape[1]//2 - 150, frame.shape[0]//2),
                                      cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 3)
                        else:
                            # Normal green box for safe distance vehicles
                            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        
                        label = f"{self.vehicle_classes[cls]} {conf:.2f}"
                        cv2.putText(frame, label, (x1, y1-10), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                        
                        # License plate detection
                        plate_region = frame[y1:y2, x1:x2]
                        plate_text = self.detect_license_plate(plate_region)
                        if plate_text:
                            cv2.putText(frame, f"Plate: {plate_text}", (x1, y2+20), 
                                      cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
                    
                    else:  # Traffic signal
                        color = self.detect_traffic_light_color(frame, (x1, y1, x2, y2))
                        signal_color = (0, 0, 255) if color == "red" else (0, 255, 0) if color == "green" else (0, 255, 255)
                        cv2.rectangle(frame, (x1, y1), (x2, y2), signal_color, 2)
                        cv2.putText(frame, f"Signal: {color}", (x1, y1-10), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.7, signal_color, 2)
        
        # Add colored overlay based on proximity status
        overlay = frame.copy()
        if proximity_status == "STOP":
            cv2.rectangle(overlay, (0, 0), (frame.shape[1], frame.shape[0]), (0, 0, 255), -1)
            alpha = 0.3
        elif proximity_status == "SLOW":
            cv2.rectangle(overlay, (0, 0), (frame.shape[1], frame.shape[0]), (0, 255, 255), -1)
            alpha = 0.2
        
        if proximity_status in ["STOP", "SLOW"]:
            frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)
        
        # Draw status on frame
        status_color = (0, 0, 255) if "STOP" in self.current_status else (0, 255, 255) if "SLOW" in self.current_status else (0, 255, 0)
        cv2.putText(frame, f"Status: {self.current_status}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, status_color, 2)
        
        return frame, self.current_status, direction

def main():
    st.title("Vehicle Detection System")
    
    detector = VehicleDetector()
    
    video_placeholder = st.empty()
    status_placeholder = st.empty()
    direction_placeholder = st.empty()
    
    video_file = st.file_uploader("Upload Video", type=['mp4', 'avi', 'mov'])
    
    if video_file:
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(video_file.read())
        
        cap = cv2.VideoCapture(tfile.name)
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            processed_frame, status, direction = detector.process_frame(frame)
            processed_frame_rgb = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
            
            # Update video frame
            video_placeholder.image(processed_frame_rgb, channels="RGB", use_column_width=True)
            
            # Update status and direction
            col1, col2 = st.columns(2)
            with col1:
                status_color = "red" if "STOP" in status else "yellow" if "SLOW" in status else "green"
                status_placeholder.markdown(
                    f"""
                    <div style='padding: 20px; background-color: {status_color}; 
                               color: {'black' if status_color == 'yellow' else 'white'}; 
                               font-size: 24px; text-align: center; 
                               border-radius: 10px; margin: 10px 0;'>
                        Status: {status}
                    </div>
                    """,
                    unsafe_allow_html=True )
            
            with col2:
                direction_placeholder.markdown(
                    f"""
                    <div style='padding: 20px; background-color: #FFD700; 
                               color: black; font-size: 24px; text-align: center; 
                               border-radius: 10px; margin: 10px 0;'>
                        Direction: {direction.upper()}
                    </div>
                    """,
                    unsafe_allow_html=True
                )
            
            time.sleep(1/30)  # Control frame rate
            
        cap.release()
        os.unlink(tfile.name)

    # Add additional control options
    st.sidebar.header("Control Panel")
    
    # Adjust detection thresholds
    st.sidebar.subheader("Detection Thresholds")
    if 'detector' in locals():
        detector.stop_threshold = st.sidebar.slider(
            "Stop Threshold",
            min_value=0.1,
            max_value=0.8,
            value=0.4,
            help="Adjust the threshold for STOP warning"
        )
        
        detector.slow_threshold = st.sidebar.slider(
            "Slow Threshold",
            min_value=0.1,
            max_value=0.8,
            value=0.25,
            help="Adjust the threshold for SLOW warning"
        )
        
        detector.confidence_threshold = st.sidebar.slider(
            "Detection Confidence",
            min_value=0.1,
            max_value=1.0,
            value=0.5,
            help="Adjust the confidence threshold for object detection"
        )
    
    # Add information section
    st.sidebar.subheader("Information")
    st.sidebar.info("""
    This system detects:
    - Vehicles (Cars, Motorcycles, Buses, Trucks)
    - Traffic Signals
    - Lane Lines
    - License Plates
    
    Status Indicators:
    🟢 GO - Safe distance
    🟡 SLOW - Moderate distance
    🔴 STOP - Too close
    """)
    
    # Add error handling section
    error_placeholder = st.empty()
    try:
        pass  # Your error handling code here
    except Exception as e:
        error_placeholder.error(f"An error occurred: {str(e)}")
        
    # Add usage instructions
    st.markdown("""
    ### Instructions:
    1. Upload a video file (MP4, AVI, or MOV format)
    2. The system will automatically detect:
        - Vehicles and their proximity
        - Traffic signals and their colors
        - Lane markings and suggested direction
        - License plates (when visible)
    3. Use the Control Panel on the left to adjust detection sensitivities
    
    ### Status Meanings:
    - **GO**: No immediate threats detected
    - **SLOW**: Vehicle detected at moderate distance
    - **STOP**: Vehicle too close or red traffic signal
    """)

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        st.error("Please refresh the page and try again.")