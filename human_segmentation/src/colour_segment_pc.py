from human_segmentation.src.preprocess_pc import load_point_cloud, preprocess_point_cloud
from human_segmentation.src.bbox_calculate import HumanBoundingBoxes
from human_segmentation.utils.colored_export import export_colored_pointcloud

def main():
    # Input and output paths
    input_path = "path/to/your/input_model.ply"  # Replace with your model path
    output_path = "colored_model.ply"
    
    # Load and preprocess point cloud
    print("Loading point cloud...")
    points = load_point_cloud(input_path)
    processed_points = preprocess_point_cloud(points)
    
    # Calculate bounding boxes
    print("Computing bounding boxes...")
    bbox_calculator = HumanBoundingBoxes()
    bounding_boxes = bbox_calculator.compute_bounding_boxes(processed_points)
    
    # Export colored point cloud
    print("Exporting colored point cloud...")
    export_colored_pointcloud(processed_points, bounding_boxes, output_path)
    print(f"Colored point cloud saved to: {output_path}")
    
    # Print color legend
    print("\nColor Legend:")
    color_map = {
        'Head': 'Red',
        'Torso': 'Green',
        'Left Arm': 'Blue',
        'Right Arm': 'Yellow',
        'Left Hand': 'Magenta',
        'Right Hand': 'Cyan',
        'Left Leg': 'Purple',
        'Right Leg': 'Olive',
        'Left Foot': 'Orange',
        'Right Foot': 'Light Blue'
    }
    for part, color in color_map.items():
        print(f"{part}: {color}")

if __name__ == "__main__":
    main()