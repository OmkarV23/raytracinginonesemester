import pyvista as pv
import sys
import os
import numpy as np

def visualize_obj(file_path):
    if not os.path.exists(file_path):
        print(f"Error: File {file_path} not found.")
        return

    print(f"Loading {file_path}...")
    try:
        # Read the mesh (lines)
        mesh = pv.read(file_path)
        
        # We know each AABB has 12 lines.
        # We can assign a scalar "Node ID" to each cell (line) to allow filtering.
        n_lines = mesh.n_cells
        n_boxes = n_lines // 12
        
        print(f"Detected {n_boxes} AABBs ({n_lines} lines).")

        # Create an ID array: 0 for first 12 lines, 1 for next 12, etc.
        # np.repeat repeats each element 12 times: [0,0..0, 1,1..1, ...]
        node_ids = np.repeat(np.arange(n_boxes), 12)
        
        # If the file has extra lines or incomplete boxes, trim the array or pad it
        if len(node_ids) > n_lines:
            node_ids = node_ids[:n_lines]
        elif len(node_ids) < n_lines:
            # Should not happen if generated correctly, but handle just in case
            padding = np.zeros(n_lines - len(node_ids))
            node_ids = np.concatenate([node_ids, padding])

        mesh.cell_data["Node ID"] = node_ids

        plotter = pv.Plotter()
        
        # Add the mesh. We will update the threshold dynamically or use the built-in threshold filter.
        # However, PyVista sliders usually callback a function.
        
        # Initial range: Show all
        low_idx = 0
        high_idx = n_boxes - 1

        # Calculate P (approximate) to suggest where leaves start
        # Total = 2P - 1 => P = (Total + 1) / 2
        p_est = (n_boxes + 1) // 2
        leaf_start = p_est - 1

        print(f"Estimated Internal Nodes: 0 to {leaf_start-1}")
        print(f"Estimated Leaf Nodes: {leaf_start} to {n_boxes-1}")

        # Actor reference
        actor = plotter.add_mesh(mesh, color="lime", show_edges=True, line_width=1, label="BVH Nodes")

        def update_threshold(value):
            # value is the slider value. simpler to just manage the mesh visibility via threshold
            # But add_mesh_threshold is not a direct widget.
            return

        # Alternative: Use the threshold filter directly
        # We create a thresholded copy
        thresh_mesh = mesh.threshold([low_idx, high_idx], scalars="Node ID")
        plotter.clear() # Clear the basic mesh
        plotter.add_mesh(thresh_mesh, color="lime", style='wireframe', line_width=1)
        plotter.add_axes()
        plotter.show_grid()
        plotter.add_text(f"BVH Visualization: {os.path.basename(file_path)}", position='upper_left')

        # Callback for start slider
        def set_start(val):
            nonlocal low_idx
            low_idx = int(val)
            new_mesh = mesh.threshold([low_idx, high_idx], scalars="Node ID")
            # Updating mesh in place is tricky in VTK/PyVista without removing/adding or using deeper tech.
            # Easiest way in pure python script: just replace the mesh in the plotter (might flicker slightly)
            # OR better: The threshold filter output is a mesh. If we keep a reference to a Mutable mesh?
            # Let's simple remove and add.
            plotter.clear_actors()
            plotter.add_mesh(new_mesh, color="lime", style='wireframe', line_width=1)
            
        def set_end(val):
            nonlocal high_idx
            high_idx = int(val)
            new_mesh = mesh.threshold([low_idx, high_idx], scalars="Node ID")
            plotter.clear_actors()
            plotter.add_mesh(new_mesh, color="lime", style='wireframe', line_width=1)

        # Add Sliders
        plotter.add_slider_widget(
            set_start, 
            [0, n_boxes-1], 
            value=0, 
            title="Start Node Index", 
            pointa=(0.025, 0.1), 
            pointb=(0.25, 0.1),
            style='modern'
        )
        
        plotter.add_slider_widget(
            set_end, 
            [0, n_boxes-1], 
            value=n_boxes-1, 
            title="End Node Index", 
            pointa=(0.025, 0.25), 
            pointb=(0.25, 0.25),
            style='modern'
        )

        print("Interactive Mode: Use sliders to filter Internal vs Leaf nodes.")
        plotter.show()
        
    except Exception as e:
        print(f"Error visualizing file: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    if len(sys.argv) > 1:
        target_file = sys.argv[1]
    else:
        # Default to the output file from C++
        target_file = "bvh.obj"
    
    visualize_obj(target_file)
