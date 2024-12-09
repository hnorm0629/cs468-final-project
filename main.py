import mcubes
import numpy as np
from scipy.ndimage import label, binary_dilation
from sklearn.cluster import KMeans
import time


def compute_gradient_magnitudes(gradient_array):
    grad_x, grad_y, grad_z = gradient_array
    gradient_magnitudes = np.sqrt(grad_x**2 + grad_y**2 + grad_z**2)

    min_value = gradient_magnitudes.min()
    max_value = gradient_magnitudes.max()
    print(f"Gradient Magnitude Range: Min = {min_value}, Max = {max_value}")

    return gradient_magnitudes

def threshold_magnitudes(gradient_magnitudes, threshold=None):
    if threshold is None:
        # Automatically set threshold as a fraction of the maximum gradient magnitude
        threshold = 0.05 * gradient_magnitudes.max()

    # Create a binary mask where gradient magnitude exceeds the threshold
    surface_mask = gradient_magnitudes > threshold
    return surface_mask

def enhance_connectivity(surface_mask):
    # Use binary dilation to close small gaps in the mask
    enhanced_mask = binary_dilation(surface_mask, iterations=2)
    return enhanced_mask

def compute_normals(gradient_array, gradient_magnitudes, surface_mask):
    grad_x, grad_y, grad_z = gradient_array

    # Avoid division by zero
    gradient_magnitudes[gradient_magnitudes == 0] = 1e-8

    # Normalize gradients to compute normals
    normals = np.stack([
        grad_x / gradient_magnitudes,
        grad_y / gradient_magnitudes,
        grad_z / gradient_magnitudes
    ], axis=0)

    # Mask normals to only include surface voxels
    normals[:, ~surface_mask] = 0
    return normals

def get_exterior_surface_mask(enhanced_mask):
    exterior_mask = (sdf > -0.05)
    exterior_surface_mask = enhanced_mask & exterior_mask
    return exterior_surface_mask

def cluster_surface_normals(exterior_surface_mask, normals, num_clusters):
    # Flatten normals for clustering
    surface_voxel_indices = np.argwhere(exterior_surface_mask)
    surface_normals = normals[:, exterior_surface_mask].T  # Shape: (num_surface_voxels, 3)

    # Transform normals for angular clustering (optional: project to a tangent space)
    transformed_normals = surface_normals / np.linalg.norm(surface_normals, axis=1, keepdims=True)

    # Cluster normals using K-Means
    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    labels = kmeans.fit_predict(transformed_normals)
    
    # Create labeled regions mask
    labeled_regions = np.zeros_like(exterior_surface_mask, dtype=np.int32)
    labeled_regions[tuple(surface_voxel_indices.T)] = labels + 1  # Ensure labels start from 1

    return labeled_regions

def refine_segments_with_connectivity(labeled_regions):
    unique_labels = np.unique(labeled_regions)
    unique_labels = unique_labels[unique_labels > 0]  # Exclude background (label 0)

    refined_regions = np.zeros_like(labeled_regions, dtype=np.int32)
    current_label = 1

    for label_id in unique_labels:
        # Isolate the current cluster
        cluster_mask = labeled_regions == label_id

        # Perform connected component analysis
        labeled_components, num_components = label(cluster_mask)
        
        # Find the largest connected component
        largest_component = max(
            range(1, num_components + 1),
            key=lambda x: np.sum(labeled_components == x)
        )

        # Assign a new unique label to the largest component
        refined_regions[labeled_components == largest_component] = current_label
        current_label += 1

    return refined_regions

def segment_surfaces(gradient_array, threshold=None, num_faces=10):
    # Step 1: Compute gradient magnitudes
    gradient_magnitudes = compute_gradient_magnitudes(gradient_array)
    
    # Step 2: Threshold magnitudes to identify surface
    surface_mask = threshold_magnitudes(gradient_magnitudes, threshold)
    
    # Step 3: Enhance connectivity of the surface mask
    enhanced_mask = enhance_connectivity(surface_mask)

    # Step 4: Compute surface normals for refinement
    normals = compute_normals(gradient_array, gradient_magnitudes, enhanced_mask)

    # Step 5: Identify exterior surface voxels
    exterior_surface_mask = get_exterior_surface_mask(enhanced_mask)

    # Step 6: Cluster surface normals using K-Means
    labeled_regions = cluster_surface_normals(exterior_surface_mask, normals, num_faces)

    # Step 7: Refine segments using connected component analysis
    refined_regions = refine_segments_with_connectivity(labeled_regions)

    # Optional: Count the number of unique regions
    num_regions = len(np.unique(refined_regions)) - 1 # Should equal `num_faces`
    
    return refined_regions, num_regions

def save_segmented_visualization_objs(sdf, labeled_regions, geom):
    segment_ids = np.unique(labeled_regions)
    segment_ids = segment_ids[segment_ids > 0]  # Exclude background (label 0)

    for segment_id in segment_ids:
        # Mask the original SDF to focus on this segment
        segment_mask = (labeled_regions == segment_id)
        masked_sdf = np.where(segment_mask, sdf, 1.0)  # Set non-segment regions to a high value
        
        # Apply Marching Cubes to extract the surface
        vertices, triangles = mcubes.marching_cubes(masked_sdf, 0.0)
        
        # Normalize vertex positions based on resolution
        resolution = 256
        vertices = (vertices / resolution) * 2.0 - 1.0
        triangles = triangles[:, ::-1]
        
        # Save the mesh to an OBJ file
        dir_obj_path = f"data/multiview/segments/{geom}/{geom}_segment_{segment_id}.obj"
        mcubes.export_obj(vertices, triangles, dir_obj_path)
        print(f"Saved segment {segment_id} to {dir_obj_path}")

if __name__ == "__main__":
    geoms = [("cube", 6), ("cylinder", 5), ("uv_sphere", 1), ("bottle", 5), ("bowl", 6), ("birdhouse", 23), ("table", 18), ("ico_sphere", 80)]
    geom, num_faces = geoms[3]
    data = np.load("data/multiview/sdf/" + geom + "_MV.npz")

    # Access the arrays in the `.npz` file
    print(data.files)
    sdf = data['sdf']
    gradients = data['gradients']

    start_time = time.time()

    # Segment surfaces off SDF geometry
    labeled_regions, num_regions = segment_surfaces(gradients, num_faces=num_faces)
    print(f"Number of surface segments: {num_regions}")

    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Elapsed time: {elapsed_time:.4f} seconds")
    
    # Convert segmented SDF surfaces to meshes
    save_segmented_visualization_objs(sdf, labeled_regions, geom)