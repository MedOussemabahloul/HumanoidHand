#!/usr/bin/env python3
import os
import mujoco
from mujoco import viewer as mj_viewer

def launch_combined_scene():
    # 1) Chemins vers tes fichiers XML existants
    asset_root   = "assets"
    hand_path    = os.path.join(asset_root, "hands/g1.xml")
    object_path  = os.path.join(asset_root, "objects/box/box.xml")
    table_path   = os.path.join(asset_root, "scenes/table.xml")

    # 2) Construis la scène combinée en mémoire
    combined_xml = f"""<?xml version="1.0" encoding="utf-8"?>
<mujoco model="combined_scene">
  <compiler meshdir="../meshes" angle="radian"/>
  <option timestep="0.005" gravity="0 0 -9.81" integrator="RK4"/>
  <worldbody>
    <include file="{hand_path}"/>
    <include file="{object_path}"/>
    <include file="{table_path}"/>
  </worldbody>
</mujoco>
"""

    # 3) Charge le modèle depuis la chaîne XML
    model = mujoco.MjModel.from_xml_string(combined_xml)
    data  = mujoco.MjData(model)

    # 4) Lance le viewer
    print("🖥️  Launching combined Mujoco viewer…")
    mj_viewer.launch(model, data)


if __name__ == "__main__":
    launch_combined_scene()
