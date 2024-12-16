import os
from visualize import AirbnbVisualizer

def main():
    # Use the project ID directly
    project_id = "lateral-vision-438701-u5"

    # Initialize visualizer
    visualizer = AirbnbVisualizer(project_id)

    # Generate all visualizations
    print("Generating price distribution visualization...")
    visualizer.price_distribution()

    print("Generating room type analysis...")
    visualizer.room_type_analysis()

    print("Generating location heatmap...")
    visualizer.location_heatmap()

    print("Generating amenities analysis...")
    visualizer.amenities_analysis()

    print("Generating reviews analysis...")
    visualizer.reviews_analysis()

    print("All visualizations have been generated in the 'visualizations' directory.")

if __name__ == "__main__":
    main()
