import os
import re

URDF_DIR = "assets/arctic_assets/object_urdf_mass/"

for filename in os.listdir(URDF_DIR):
    if not filename.endswith(".urdf"):
        continue
    path = os.path.join(URDF_DIR, filename)
    with open(path, 'r') as f:
        data = f.read()

    # Remplacer le chemin mesh, ne garder que le nom du fichier
    # ex: ./object_vtemplates/laptop/bottom.obj -> bottom.obj
    new_data = re.sub(r'filename="([^"/]+/)*([^"/]+\.obj)"', r'filename="\2"', data)
    # Pour .dae aussi éventuellement
    new_data = re.sub(r'filename="([^"/]+/)*([^"/]+\.dae)"', r'filename="\2"', new_data)

    with open(path, 'w') as f:
        f.write(new_data)
    print(f"Corrigé : {filename}")
