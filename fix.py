import os
import shutil
import xml.etree.ElementTree as ET

# Chemins à adapter !
URDF_SRC = "assets/arctic_assets/object_urdf_mass"
MESH_ROOT = "assets/arctic_assets/object_vtemplates"  # Dossier racine où tu as tes .obj
DEST_ROOT = "assets/objects"

os.makedirs(DEST_ROOT, exist_ok=True)

for urdf_file in os.listdir(URDF_SRC):
    if not urdf_file.endswith(".urdf"):
        continue

    obj_name = urdf_file.replace('.urdf','')
    obj_dir = os.path.join(DEST_ROOT, obj_name)
    os.makedirs(obj_dir, exist_ok=True)

    # Copie le urdf
    src_urdf = os.path.join(URDF_SRC, urdf_file)
    dst_urdf = os.path.join(obj_dir, urdf_file)

    # Corriger les chemins dans le .urdf
    tree = ET.parse(src_urdf)
    root = tree.getroot()

    for mesh in root.iter('mesh'):
        mesh_path = mesh.attrib.get('filename')
        if mesh_path is None:
            continue

        mesh_clean = mesh_path.lstrip('./')
        mesh_file = os.path.basename(mesh_clean)
        # Cherche le fichier mesh dans le MESH_ROOT
        # On suppose l'organisation: object_vtemplates/<objet>/<meshfile>
        found = False
        for dirpath, _, filenames in os.walk(MESH_ROOT):
            if mesh_file in filenames:
                src_mesh = os.path.join(dirpath, mesh_file)
                dst_mesh = os.path.join(obj_dir, mesh_file)
                shutil.copy2(src_mesh, dst_mesh)
                mesh.set('filename', mesh_file)
                found = True
                break
        if not found:
            print(f"[WARNING] mesh {mesh_file} not found for {urdf_file} !")

    tree.write(dst_urdf)
    print(f"Obj {obj_name} ready in {obj_dir}")
