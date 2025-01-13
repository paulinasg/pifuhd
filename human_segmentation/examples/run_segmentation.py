import os
import sys
import numpy as np
import open3d as o3d

# Add the src directory to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from bbox_calculate import BoundingBoxCalculator
from preprocess_pc import preprocess_point_cloud

def main():
    # Set up paths
    input_path = os.path.join('input_models', 'man_pc_compact.ply')
    output_dir = 'results'
    output_path = os.path.join(output_dir, 'segmented_model_3.ply')
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        # Load point cloud
        print(f"Loading point cloud from: {input_path}")
        pcd = o3d.io.read_point_cloud(input_path)
        points = np.asarray(pcd.points)
        
        # Preprocess point cloud
        print("Preprocessing point cloud...")
        processed_points = preprocess_point_cloud(points)
        
        # Calculate bounding boxes
        print("Calculating bounding boxes...")
        calculator = BoundingBoxCalculator()
        bboxes = calculator.calculate_bboxes(processed_points)
        
        # Print bounding box information
        print("\nBounding Box Information:")
        for part_name, bbox in bboxes.items():
            if bbox is not None:
                print(f"\n{part_name.upper()}:")
                print(f"  Dimensions (w,h,d): {[round(d, 3) for d in bbox['dimensions']]}")
                print(f"  Center point: {[round(c, 3) for c in bbox['center']]}")
        
        # Save colored point cloud
        print(f"\nSaving colored point cloud to: {output_path}")
        calculator.save_colored_ply(processed_points, output_path)
        
        print("\nColor Legend:")
        color_names = {
            'head': 'Red',
            'torso': 'Green',
            'left_arm': 'Blue',
            'right_arm': 'Yellow',
            'left_hand': 'Magenta',
            'right_hand': 'Cyan',
            'left_leg': 'Purple',
            'right_leg': 'Olive',
            'left_foot': 'Orange',
            'right_foot': 'Light Blue'
        }
        for part, color in color_names.items():
            print(f"{part}: {color}")
            
        print("\nSegmentation completed successfully!")
        
    except Exception as e:
        print(f"Error during processing: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()