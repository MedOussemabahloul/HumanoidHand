import os
import xml.etree.ElementTree as ET

def find_multiple_roots(folder):
    for dirpath, _, filenames in os.walk(folder):
        for fname in filenames:
            if fname.endswith('.urdf'):
                path = os.path.join(dirpath, fname)
                try:
                    tree = ET.parse(path)
                    root = tree.getroot()
                    links = [l.attrib['name'] for l in root.findall('.//link')]
                    joints = [j for j in root.findall('.//joint')]
                    parents = set(j.find('parent').attrib['link'] for j in joints if j.find('parent') is not None)
                    roots = set(links) - parents
                    if len(roots) > 1:
                        print(f"[MULTIROOT] {path} => {roots}")
                except Exception as e:
                    print(f"[BAD XML] {path} : {e}")

find_multiple_roots("assets/hands")
