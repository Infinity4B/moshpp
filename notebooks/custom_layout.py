from moshpp.marker_layout.create_marker_layout_for_mocaps import marker_labels_to_marker_layout

markers=["R.Shoulder", "R.Offset", "R.Elbow", "R.Wrist", "L.Shoulder", "L.Elbow", "L.Wrist", "R.ASIS", "L.ASIS", "V.Sacral", "R.Thigh", "R.Knee", "R.Shank", "R.Ankle", "R.Heel", "R.Toe", "L.Thigh", "L.Knee", "L.Shank", "L.Ankle", "L.Heel", "L.Toe"]
marker_labels_to_marker_layout(chosen_markers=markers,marker_layout_fname='./ours.json',surface_model_type='smplx')