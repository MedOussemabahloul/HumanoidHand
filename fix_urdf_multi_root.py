import os
import xml.etree.ElementTree as ET

def fix_multi_root_urdf(urdf_path):
    try:
        tree = ET.parse(urdf_path)
        root = tree.getroot()
    except Exception as e:
        print(f"[ERR] Can't parse: {urdf_path} -- {e}")
        return

    link_names = set(link.attrib['name'] for link in root.findall('.//link'))
    child_links = set(joint.find('child').attrib['link'] for joint in root.findall('.//joint') if joint.find('child') is not None)
    root_links = list(link_names - child_links)

    if len(root_links) <= 1:
        print(f"[OK] {urdf_path}: only one root.")
        return

    true_root = root_links[0]
    for other in root_links[1:]:
        joint = ET.SubElement(root, 'joint', {'name': f'auto_fixed_{other}', 'type': 'fixed'})
        ET.SubElement(joint, 'parent', {'link': true_root})
        ET.SubElement(joint, 'child', {'link': other})
    new_path = urdf_path.replace(".urdf", "_fixed.urdf")
    tree.write(new_path)
    print(f"[FIXED] {urdf_path} -> {new_path}")

def fix_folder(folder):
    for dirpath, _, files in os.walk(folder):
        for f in files:
            if f.endswith('.urdf'):
                fix_multi_root_urdf(os.path.join(dirpath, f))

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python fix_urdf_multi_root.py <urdf_folder>")
    else:
        fix_folder(sys.argv[1])
