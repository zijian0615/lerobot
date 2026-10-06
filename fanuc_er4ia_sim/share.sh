#!/usr/bin/env bash
# Copy the shareable file set into a directory (default ./fanuc_er4ia_sim_share).
# Excludes the venv, generated textures, caches and rendered video.
set -euo pipefail
DEST="${1:-fanuc_er4ia_sim_share}"
SRC="$(cd "$(dirname "$0")" && pwd)"
mkdir -p "$DEST" "$DEST/run" "$DEST/reference" "$DEST/assets"

# scene + simulation
cp "$SRC"/{scene_config.py,cell.py,fk.py,ik.py,build_scene.py} "$DEST"/
cp "$SRC"/{encode_video.py,inspect_gripper.py,verify_fk.py,make_textures.py} "$DEST"/
cp "$SRC"/{render_settings.py,test_physics.py,test_bin_physics.py} "$DEST"/

# the real mat artwork, rectified from the site scan (recipients do not have
# the scan, so the texture itself is shipped; make_mat_texture.py rebuilds it)
cp "$SRC"/assets/mat_print.png "$DEST/assets/"
cp "$SRC"/{make_mat_texture.py,import_assets.py,solve_tablecam_scan.py} "$DEST"/

# perception pipeline
cp "$SRC"/{capture_scene.py,perceive.py,backproject.py,plan_grasps.py} "$DEST"/
cp "$SRC"/{execute_plan.py,run_pipeline.sh} "$DEST"/
chmod +x "$DEST"/run_pipeline.sh

# calibration reference
cp "$SRC"/{extract_lines.py,fit_distortion.py,solve_focal.py,refit_push.py} "$DEST"/
cp "$SRC"/{transfer_candidates.py,calib_plate.png,calib_plate_push.png} "$DEST"/

# docs + setup
cp "$SRC"/{requirements.txt,.gitignore,README.md,CALIBRATION.md,share.sh} "$DEST"/
cp "$SRC"/{scene_closeup.png,gripper_material.png} "$DEST"/ 2>/dev/null || true

# a worked example of the perception pipeline, minus the big video files
for f in world_camera.png detections.json detections_overlay.png plan.json \
         verify.json final_frame.png camera.json ground_truth.json \
         bin_view.png inventory.json; do
    cp "$SRC/run/$f" "$DEST/run/" 2>/dev/null || true
done

# stills of the physical cell, for anyone reconciling the model against reality
cp "$SRC"/reference/*.jpg "$DEST/reference/" 2>/dev/null || true
cp "$SRC"/reference/*.png "$DEST/reference/" 2>/dev/null || true

echo "wrote $(find "$DEST" -type f | wc -l) files to $DEST ($(du -sh "$DEST" | cut -f1))"
echo "Recipient: pip install -r requirements.txt && python make_textures.py"
