# %%
# using FreeCAD to read BRep files and extract surface vertex coordinates: "freecadcmd data_process_brep.py"
import os
import FreeCAD
import Part
import Mesh
import numpy as np
import time

# SCRIPT_PATH = os.path.dirname(os.path.abspath(__file__))
SCRIPT_PATH = "/work/hdd/bdsy/qibang/repository_Wbdsy/GINOT/data_scripts/GEJetEngineBracket"
data_path = f"{SCRIPT_PATH}/../../data/GEJetEngineBracket/"
brep_path = f"{data_path}/BRep/"
# %%


def extract_vertices_from_step(step_file):
    """Load a STEP file and extract vertex coordinates."""
    shape = Part.read(step_file)
    vertices = [v.Point for v in shape.Vertexes]
    return vertices


def process_step_files(input_folder):
    points_cloud_dict = {}
    """Process all STEP files in a folder and save vertex data."""
    brep_files = os.listdir(input_folder)
    sample_ids = [os.path.splitext(f)[0] for f in brep_files]
    time_start = time.time()
    for i, (filename, s_id) in enumerate(zip(brep_files, sample_ids)):
        step_path = os.path.join(input_folder, filename)
        vertices = extract_vertices_from_step(step_path)
        points_cloud_dict[s_id] = vertices
        print(f"Processed {filename}: Extracted {len(vertices)} vertices")
        print(
            f"Time elapsed: {time.time() - time_start:.2f} s for {i+1}/{len(sample_ids)} files")
    return points_cloud_dict


# Set your input and output folders

# Run the batch process
points_cloud_dict = process_step_files(brep_path)
print("Processing complete!")
output_file = os.path.join(data_path, "points_cloud_dict.npz")
np.savez(output_file, **points_cloud_dict)
print(f"Saved points cloud data to {output_file}")
