import os
from pathlib import Path
import xml.etree.ElementTree as ET
import logging

# 📍 Configuration
SOURCE_DIR = Path("assets/hands")
TARGET_DIR = Path("assets/hands_mjcf")
TARGET_DIR.mkdir(parents=True, exist_ok=True)
LOGS = []

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")

def is_urdf_like(path: Path) -> bool:
    try:
        tree = ET.parse(path)
        root = tree.getroot()
        tags = {el.tag for el in root.iter()}
        return 'link' in tags and 'joint' in tags
    except ET.ParseError:
        return False

def convert_urdf_like(input_path: Path, output_path: Path) -> bool:
    try:
        tree = ET.parse(input_path)
        root = tree.getroot()

        mujoco = ET.Element("mujoco", model=input_path.stem)
        ET.SubElement(mujoco, "compiler", meshdir="meshes", angle="radian")
        ET.SubElement(mujoco, "option", timestep="0.005", gravity="0 0 -9.81", integrator="RK4")
        worldbody = ET.SubElement(mujoco, "worldbody")

        for elem in root:
            if elem.tag in ("link", "joint"):
                worldbody.append(elem)

        output_path.parent.mkdir(parents=True, exist_ok=True)
        tree_out = ET.ElementTree(mujoco)
        ET.indent(tree_out, space="  ", level=0)
        tree_out.write(output_path, encoding="utf-8", xml_declaration=True)

        logging.info(f"✅ Converti : {input_path} → {output_path}")
        LOGS.append(f"✅ {input_path} → {output_path}")
        return True
    except Exception as e:
        logging.error(f"❌ Erreur sur {input_path}: {e}")
        LOGS.append(f"❌ {input_path}: {e}")
        return False

def scan_and_convert_all():
    total_files = 0
    converted = 0
    skipped = 0

    for file_path in SOURCE_DIR.rglob("*"):
        if file_path.suffix.lower() in (".xml", ".urdf") and file_path.is_file():
            total_files += 1
            if is_urdf_like(file_path):
                relative = file_path.relative_to(SOURCE_DIR)
                output_name = relative.stem + "_mjcf.xml"
                output_path = TARGET_DIR / relative.parent / output_name
                success = convert_urdf_like(file_path, output_path)
                converted += int(success)
            else:
                skipped += 1
                logging.info(f"⚪ Ignoré (non URDF-like) : {file_path}")

    # ✍️ Rapport
    report_path = TARGET_DIR / "conversion_report.txt"
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("=== Rapport de conversion URDF → MJCF ===\n\n")
        f.write(f"Total fichiers scannés : {total_files}\n")
        f.write(f"Fichiers convertis     : {converted}\n")
        f.write(f"Fichiers ignorés       : {skipped}\n\n")
        f.write("=== Détails ===\n")
        for line in LOGS:
            f.write(line + "\n")

    logging.info(f"📄 Rapport généré : {report_path}")

if __name__ == "__main__":
    scan_and_convert_all()
