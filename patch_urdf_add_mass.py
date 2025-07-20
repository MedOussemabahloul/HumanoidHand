import os
import re
import xml.etree.ElementTree as ET
from shutil import copy2

DEFAULT_MASS = 1.0
DEFAULT_INERTIA = {
    "ixx": 0.001, "ixy": 0.0, "ixz": 0.0,
    "iyy": 0.001, "iyz": 0.0, "izz": 0.001
}
DEFAULT_ORIGIN = 'xyz="0 0 0" rpy="0 0 0"'

def build_inertial_block():
    inertia_str = ' '.join([f'{k}="{v}"' for k, v in DEFAULT_INERTIA.items()])
    return f"""
    <inertial>
      <origin {DEFAULT_ORIGIN}/>
      <mass value="{DEFAULT_MASS}"/>
      <inertia {inertia_str}/>
    </inertial>"""

def patch_urdf_file(filepath, outpath):
    # On lit le fichier URDF
    try:
        with open(filepath, 'r') as f:
            content = f.read()
    except Exception as e:
        print(f"[ERREUR] Impossible de lire {filepath}: {e}")
        return False

    content = content.replace('&', '&amp;')  # quick fix XML
    try:
        root = ET.fromstring(content)
    except ET.ParseError as e:
        print(f"[ERREUR] XML invalide dans {filepath} : {e}")
        return False

    # Cherche tous les links et joints
    links = [l for l in root.findall('.//link')]
    joints = [j for j in root.findall('.//joint')]
    link_names = [l.attrib['name'] for l in links]
    parent_links = set(j.find('child').attrib['link'] for j in joints if j.find('child') is not None)
    child_links = set(j.find('parent').attrib['link'] for j in joints if j.find('parent') is not None)
    root_links = set(link_names) - parent_links

    # 1. Correction multi-root
    if len(root_links) > 1:
        print(f"[PATCH] {filepath} => MULTIROOT ({root_links}), ajout d'un virtual_root.")
        # Ajoute un lien racine virtuel
        virtual_link = ET.Element('link', {'name': 'virtual_root'})
        # Inertial pour virtual_root (optionnel)
        inertial = ET.fromstring(build_inertial_block())
        virtual_link.append(inertial)
        root.insert(0, virtual_link)
        # Ajoute un joint fixe pour chaque lien racine existant
        for root_name in root_links:
            joint = ET.Element('joint', {'name': f'{root_name}_to_virtual_root', 'type': 'fixed'})
            ET.SubElement(joint, 'parent', {'link': 'virtual_root'})
            ET.SubElement(joint, 'child', {'link': root_name})
            root.insert(1, joint)

    # 2. Ajoute inertial si absent
    for link in root.findall('.//link'):
        if link.find('inertial') is None:
            inertial = ET.fromstring(build_inertial_block())
            link.insert(0, inertial)
        # Dummy visual si absent (optionnel, surtout si urdf2mjcf rouspète sinon)
        if link.find('visual') is None and link.find('collision') is None:
            visual = ET.SubElement(link, 'visual')
            geometry = ET.SubElement(visual, 'geometry')
            box = ET.SubElement(geometry, 'box')
            box.attrib['size'] = "0.01 0.01 0.01"

    # 3. Retire <density>
    for density in root.findall('.//density'):
        parent = density.getparent()
        if parent is not None:
            parent.remove(density)

    # 4. Écriture du nouveau fichier (remplace l’original)
    patched_content = ET.tostring(root, encoding='unicode')
    patched_content = re.sub(r'(<\?xml[^>]+\?>)?', '<?xml version="1.0"?>\n', patched_content, 1)
    with open(outpath, 'w') as f:
        f.write(patched_content)
    print(f"[OK] Patché : {outpath}")
    return True

def patch_urdf_tree(root_folder):
    for dirpath, _, filenames in os.walk(root_folder):
        for fname in filenames:
            if fname.endswith('.urdf'):
                fpath = os.path.join(dirpath, fname)
                patch_urdf_file(fpath, fpath)  # écrase le fichier original

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python patch_urdf_cinematic.py <urdf_folder>")
        sys.exit(1)
    patch_urdf_tree(sys.argv[1])
