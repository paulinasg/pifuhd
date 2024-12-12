import pymeshlab

# Load the OBJ file
ms = pymeshlab.MeshSet()
ms.load_new_mesh("/afs/inf.ed.ac.uk/user/s21/s2150959/pifuhd/pointcloud/dataset_samples/rp_posed_00178_29.obj")

# Compute a point cloud from the mesh
ms.generate_sampling_poisson_disk(samplenum=100000)

# Save the point cloud
#ms.save_current_mesh("pifuhd_pc.ply", binary=False)
ms.save_current_mesh("results/dataset_pc.ply", binary=False)