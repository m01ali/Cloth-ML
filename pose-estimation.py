import cv2
import mediapipe as mp
import numpy as np
import json
import uuid
import os
from typing import Dict, List, Tuple, Optional

class PoseEstimator:
    """
    Pose estimation component for StableVITON workflow.
    Extracts 2D pose keypoints and generates visual pose representation.
    """
    
    def __init__(self, confidence_threshold: float = 0.5, output_dir: str = "./data"):
        """
        Initialize the pose estimator.
        
        Args:
            confidence_threshold: Minimum confidence for pose detection
            output_dir: Directory to save output files
        """
        self.confidence_threshold = confidence_threshold
        self.output_dir = output_dir
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Initialize MediaPipe Pose
        self.mp_pose = mp.solutions.pose
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        
        # Initialize pose detector with high accuracy model
        self.pose = self.mp_pose.Pose(
            static_image_mode=True,
            model_complexity=2,  # Use highest accuracy model
            enable_segmentation=False,
            min_detection_confidence=confidence_threshold,
            min_tracking_confidence=confidence_threshold
        )
        
        # Define pose connections for visualization
        self.pose_connections = self.mp_pose.POSE_CONNECTIONS
        
    def extract_pose_keypoints(self, image_path: str) -> Optional[Dict]:
        """
        Extract pose keypoints from an image.
        
        Args:
            image_path: Path to the input image
            
        Returns:
            Dictionary containing pose keypoints and metadata, or None if no pose detected
        """
        # Load and preprocess image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not load image from {image_path}")
        
        # Convert BGR to RGB for MediaPipe
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Process the image
        results = self.pose.process(image_rgb)
        
        if not results.pose_landmarks:
            print("No pose detected in the image")
            return None
        
        # Extract keypoints with confidence scores
        keypoints = []
        landmarks = results.pose_landmarks.landmark
        
        # MediaPipe pose landmarks (33 points)
        landmark_names = [
            'nose', 'left_eye_inner', 'left_eye', 'left_eye_outer',
            'right_eye_inner', 'right_eye', 'right_eye_outer',
            'left_ear', 'right_ear', 'mouth_left', 'mouth_right',
            'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
            'left_wrist', 'right_wrist', 'left_pinky', 'right_pinky',
            'left_index', 'right_index', 'left_thumb', 'right_thumb',
            'left_hip', 'right_hip', 'left_knee', 'right_knee',
            'left_ankle', 'right_ankle', 'left_heel', 'right_heel',
            'left_foot_index', 'right_foot_index'
        ]
        
        # Get image dimensions
        height, width = image.shape[:2]
        
        for i, (landmark, name) in enumerate(zip(landmarks, landmark_names)):
            # Convert normalized coordinates to pixel coordinates
            x = int(landmark.x * width)
            y = int(landmark.y * height)
            confidence = landmark.visibility
            
            keypoints.append({
                'id': i,
                'name': name,
                'x': x,
                'y': y,
                'confidence': confidence
            })
        
        # Create structured pose data
        pose_data = {
            'keypoints': keypoints,
            'image_dimensions': {
                'width': width,
                'height': height
            },
            'model_info': {
                'model': 'MediaPipe Pose',
                'complexity': 2,
                'confidence_threshold': self.confidence_threshold
            },
            'total_keypoints': len(keypoints),
            'detection_confidence': min([kp['confidence'] for kp in keypoints])
        }
        
        return pose_data
    
    def create_pose_visualization(self, image_path: str, pose_data: Dict) -> np.ndarray:
        """
        Create a visual representation of the pose.
        
        Args:
            image_path: Path to the original image
            pose_data: Pose keypoints data
            
        Returns:
            Pose visualization image as numpy array
        """
        # Load original image
        image = cv2.imread(image_path)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Process image again to get MediaPipe landmarks for drawing
        results = self.pose.process(image_rgb)
        
        # Create a copy for visualization
        pose_image = image.copy()
        
        if results.pose_landmarks:
            # Draw pose landmarks and connections
            self.mp_drawing.draw_landmarks(
                pose_image,
                results.pose_landmarks,
                self.pose_connections,
                landmark_drawing_spec=self.mp_drawing_styles.get_default_pose_landmarks_style()
            )
        
        return pose_image
    
    def create_skeleton_only_visualization(self, pose_data: Dict) -> np.ndarray:
        """
        Create a skeleton-only visualization on transparent/plain background.
        
        Args:
            pose_data: Pose keypoints data
            
        Returns:
            Skeleton visualization image as numpy array
        """
        width = pose_data['image_dimensions']['width']
        height = pose_data['image_dimensions']['height']
        
        # Create blank canvas
        skeleton_image = np.zeros((height, width, 3), dtype=np.uint8)
        
        # Extract keypoint coordinates
        keypoints = pose_data['keypoints']
        points = {}
        
        for kp in keypoints:
            if kp['confidence'] > self.confidence_threshold:
                points[kp['name']] = (kp['x'], kp['y'])
        
        # Define skeleton connections
        connections = [
            # Head connections
            ('left_ear', 'left_eye_outer'),
            ('left_eye_outer', 'left_eye'),
            ('left_eye', 'left_eye_inner'),
            ('left_eye_inner', 'nose'),
            ('nose', 'right_eye_inner'),
            ('right_eye_inner', 'right_eye'),
            ('right_eye', 'right_eye_outer'),
            ('right_eye_outer', 'right_ear'),
            
            # Body connections
            ('left_shoulder', 'right_shoulder'),
            ('left_shoulder', 'left_elbow'),
            ('left_elbow', 'left_wrist'),
            ('right_shoulder', 'right_elbow'),
            ('right_elbow', 'right_wrist'),
            ('left_shoulder', 'left_hip'),
            ('right_shoulder', 'right_hip'),
            ('left_hip', 'right_hip'),
            
            # Leg connections
            ('left_hip', 'left_knee'),
            ('left_knee', 'left_ankle'),
            ('right_hip', 'right_knee'),
            ('right_knee', 'right_ankle'),
            
            # Foot connections
            ('left_ankle', 'left_heel'),
            ('left_heel', 'left_foot_index'),
            ('right_ankle', 'right_heel'),
            ('right_heel', 'right_foot_index')
        ]
        
        # Draw skeleton connections
        for connection in connections:
            point1_name, point2_name = connection
            if point1_name in points and point2_name in points:
                pt1 = points[point1_name]
                pt2 = points[point2_name]
                cv2.line(skeleton_image, pt1, pt2, (255, 255, 255), 2)
        
        # Draw keypoints
        for point_name, point_coords in points.items():
            cv2.circle(skeleton_image, point_coords, 4, (0, 255, 0), -1)
        
        return skeleton_image
    
    def process_image(self, image_path: str) -> Tuple[str, str]:
        """
        Complete pose processing pipeline for a single image.
        
        Args:
            image_path: Path to the input image
            
        Returns:
            Tuple of (pose_keypoints_file_id, pose_file_id)
        """
        print(f"Processing pose estimation for: {image_path}")
        
        # Extract pose keypoints
        pose_data = self.extract_pose_keypoints(image_path)
        
        if pose_data is None:
            raise ValueError("No pose detected in the image")
        
        # Generate unique UUIDs for output files
        pose_uuid = str(uuid.uuid4())
        pose_keypoints_filename = f"{pose_uuid}.json"
        pose_image_filename = f"{pose_uuid}.png"
        
        # Save pose keypoints JSON
        keypoints_path = os.path.join(self.output_dir, pose_keypoints_filename)
        with open(keypoints_path, 'w') as f:
            json.dump(pose_data, f, indent=2)
        
        # Create and save pose visualization
        pose_visualization = self.create_pose_visualization(image_path, pose_data)
        pose_image_path = os.path.join(self.output_dir, pose_image_filename)
        cv2.imwrite(pose_image_path, pose_visualization)
        
        print(f"Pose keypoints saved to: {keypoints_path}")
        print(f"Pose visualization saved to: {pose_image_path}")
        print(f"Detected {pose_data['total_keypoints']} keypoints with confidence >= {self.confidence_threshold}")
        
        return pose_keypoints_filename, pose_image_filename
    
    def batch_process(self, image_paths: List[str]) -> List[Tuple[str, str, str]]:
        """
        Process multiple images for pose estimation.
        
        Args:
            image_paths: List of paths to input images
            
        Returns:
            List of tuples (original_image_path, pose_keypoints_file_id, pose_file_id)
        """
        results = []
        
        for image_path in image_paths:
            try:
                pose_keypoints_id, pose_file_id = self.process_image(image_path)
                results.append((image_path, pose_keypoints_id, pose_file_id))
                print(f"✓ Successfully processed: {image_path}")
            except Exception as e:
                print(f"✗ Failed to process {image_path}: {str(e)}")
                results.append((image_path, None, None))
        
        return results
    
    def get_pose_summary(self, pose_keypoints_file: str) -> Dict:
        """
        Get a summary of pose data from a keypoints file.
        
        Args:
            pose_keypoints_file: Path to the pose keypoints JSON file
            
        Returns:
            Summary dictionary
        """
        with open(os.path.join(self.output_dir, pose_keypoints_file), 'r') as f:
            pose_data = json.load(f)
        
        high_confidence_points = [
            kp for kp in pose_data['keypoints'] 
            if kp['confidence'] > self.confidence_threshold
        ]
        
        return {
            'total_keypoints': pose_data['total_keypoints'],
            'high_confidence_keypoints': len(high_confidence_points),
            'detection_confidence': pose_data['detection_confidence'],
            'image_dimensions': pose_data['image_dimensions'],
            'key_body_parts_detected': {
                'shoulders': any(kp['name'] in ['left_shoulder', 'right_shoulder'] 
                               for kp in high_confidence_points),
                'elbows': any(kp['name'] in ['left_elbow', 'right_elbow'] 
                             for kp in high_confidence_points),
                'wrists': any(kp['name'] in ['left_wrist', 'right_wrist'] 
                             for kp in high_confidence_points),
                'hips': any(kp['name'] in ['left_hip', 'right_hip'] 
                           for kp in high_confidence_points),
                'knees': any(kp['name'] in ['left_knee', 'right_knee'] 
                            for kp in high_confidence_points)
            }
        }
    
    def __del__(self):
        """Cleanup MediaPipe resources."""
        if hasattr(self, 'pose'):
            self.pose.close()


# Example usage and testing functions
def main():
    """
    Example usage of the PoseEstimator class.
    """
    # Initialize pose estimator
    pose_estimator = PoseEstimator(
        confidence_threshold=0.5,
        output_dir="./data"
    )
    
    # Example: Process a single image
    try:
        image_path = "E:\Cloth-ML\Test-image.jpg"
        
        # Check if image exists (replace with actual image path)
        if not os.path.exists(image_path):
            print(f"Please provide a valid image path. Current path '{image_path}' does not exist.")
            print("Usage example:")
            print("1. Replace 'image_path' with your actual image file")
            print("2. Run: pose_keypoints_id, pose_image_id = pose_estimator.process_image('your_image.jpg')")
            return
        
        # Process the image
        pose_keypoints_id, pose_image_id = pose_estimator.process_image(image_path)
        
        # Get pose summary
        summary = pose_estimator.get_pose_summary(pose_keypoints_id)
        print("\nPose Summary:")
        print(f"- Total keypoints: {summary['total_keypoints']}")
        print(f"- High confidence keypoints: {summary['high_confidence_keypoints']}")
        print(f"- Detection confidence: {summary['detection_confidence']:.3f}")
        print(f"- Key body parts detected: {summary['key_body_parts_detected']}")
        
        print(f"\nOutput files:")
        print(f"- Pose keypoints: {pose_keypoints_id}")
        print(f"- Pose visualization: {pose_image_id}")
        
    except Exception as e:
        print(f"Error processing image: {str(e)}")


if __name__ == "__main__":
    main()