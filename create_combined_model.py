
#!/usr/bin/env python3
"""
Script professionnel pour créer le modèle G1 combiné (corps  doigts)
Corrige automatiquement les chemins des meshes et les attributs frictionloss
Auteur: Assistant IA
Projet: G1 Fingers Manipulation
"""

import xml.etree.ElementTree as ET
import os
import sys
from pathlib import Path
import shutil

class G1ModelBuilder:
    """Constructeur professionnel du modèle G1 combiné"""
    
    def __init__(self, project_root="/home/oussema/Documents/project"):
        self.project_root = Path(project_root)
        self.assets_dir = self.project_root / "assets" / "hands"
        self.meshes_dir = self.assets_dir / "meshes"
        self.results_dir = self.project_root / "results"
        
        # Créer le dossier results s'il n'existe pas
        self.results_dir.mkdir(exist_ok=True)
        
        # Fichiers source
        self.body_file = self.assets_dir / "g1_body.xml"
        self.fingers_file = self.assets_dir / "g1_fingers.xml"
        
        # Fichier de sortie
        self.output_file = self.results_dir / "g1_combined.xml"
    
    def verify_files(self):
        """Vérifie que tous les fichiers nécessaires existent"""
        print("🔍 Vérification des fichiers...")
        
        # Vérifier les fichiers XML
        if not self.body_file.exists():
            raise FileNotFoundError(f"Fichier manquant: {self.body_file}")
        if not self.fingers_file.exists():
            raise FileNotFoundError(f"Fichier manquant: {self.fingers_file}")
        
        # Vérifier les meshes
        if not self.meshes_dir.exists():
            raise FileNotFoundError(f"Dossier meshes manquant: {self.meshes_dir}")
        
        mesh_files = list(self.meshes_dir.glob("*.STL"))
        if not mesh_files:
            raise FileNotFoundError(f"Aucun fichier mesh (.STL) trouvé dans: {self.meshes_dir}")
        
        print(f"✅ Fichiers XML trouvés: g1_body.xml, g1_fingers.xml")
        print(f"✅ {len(mesh_files)} fichiers mesh trouvés")
    
    def fix_mesh_paths(self, xml_file):
        """Corrige les chemins des meshes dans un fichier XML"""
        print(f"🔧 Correction des chemins mesh dans: {xml_file.name}")
        

        
        # Parser et corriger
        tree = ET.parse(xml_file)
        root = tree.getroot()
        corrections = 0
        
        for mesh in root.findall(".//mesh"):
            file_attr = mesh.get("file")
            if file_attr and ("/content/" in file_attr or "/home/" in file_attr):
                mesh_name = Path(file_attr).name
                new_path = f"meshes/{mesh_name}"
                mesh.set("file", new_path)
                corrections = 1
        
        if corrections > 0:
            ET.indent(tree, space="  ", level=0)
            tree.write(xml_file, encoding="utf-8", xml_declaration=True)
            print(f"  ✅ {corrections} chemins corrigés")
        else:
            print(f"  ℹ️  Aucun chemin à corriger")
    
    def fix_frictionloss(self, xml_file):
        """Corrige les attributs frictionloss"""
        print(f"🔧 Correction frictionloss dans: {xml_file.name}")
        
        tree = ET.parse(xml_file)
        root = tree.getroot()
        
        # Éléments autorisés pour frictionloss
        allowed_elements = {'joint', 'tendon', 'spatial', 'fixed'}
        
        removed = 0
        added = 0
        
        # Supprimer frictionloss des éléments non autorisés
        for element in root.iter():
            if 'frictionloss' in element.attrib and element.tag not in allowed_elements:
                del element.attrib['frictionloss']
                removed = 1
        
        # Ajouter frictionloss aux joints qui n'en ont pas
        for joint in root.findall('.//joint'):
            if 'frictionloss' not in joint.attrib:
                joint.set('frictionloss', '0.01')
                added = 1
        
        if removed > 0 or added > 0:
            ET.indent(tree, space="  ", level=0)
            tree.write(xml_file, encoding="utf-8", xml_declaration=True)
            print(f"  ✅ {removed} supprimés, {added} ajoutés")
        else:
            print(f"  ℹ️  Aucune correction nécessaire")
    
    def create_combined_model(self):
        """Crée le modèle combiné avec environnement de manipulation"""
        print("🚀 Création du modèle combiné...")
        
        # Lire les fichiers source
        body_tree = ET.parse(self.body_file)
        fingers_tree = ET.parse(self.fingers_file)
        
        # Créer le modèle racine
        root = ET.Element("mujoco", model="g1_manipulation")
        
        # Compiler les options
        compiler = ET.SubElement(root, "compiler", angle="radian", coordinate="local")
        
        # Options de simulation optimisées
        option = ET.SubElement(root, "option", 
                              timestep="0.002", 
                              iterations="50", 
                              solver="Newton",
                              tolerance="1e-10")
        flag = ET.SubElement(option, "flag", warmstart="enable", energy="enable")
        
        # Tailles optimisées
        size = ET.SubElement(root, "size", 
                            nconmax="100", 
                            njmax="1000", 
                            nstack="600000")
        
        # Assets - Inclure les modèles
        asset = ET.SubElement(root, "asset")
        
        # Textures pour l'environnement
        tex_grid = ET.SubElement(asset, "texture", 
                                name="grid", type="2d", 
                                builtin="checker", 
                                rgb1="0.1 0.2 0.3", 
                                rgb2="0.2 0.3 0.4", 
                                width="300", height="300")
        
        mat_grid = ET.SubElement(asset, "material", 
                                name="grid", texture="grid", 
                                texrepeat="8 8", reflectance="0.2")
        
        # Worldbody avec environnement
        worldbody = ET.SubElement(root, "worldbody")
        
        # Éclairage
        light = ET.SubElement(worldbody, "light", 
                             cutoff="100", diffuse="1 1 1", 
                             dir="-0 0 -1.3", directional="true", 
                             exponent="1", pos="0 0 1.3", specular=".1 .1 .1")
        
        # Sol
        floor = ET.SubElement(worldbody, "geom", 
                             name="floor", 
                             size="2 2 0.1", 
                             type="plane", 
                             material="grid")
        
        # Table de manipulation
        table = ET.SubElement(worldbody, "body", name="table", pos="0.5 0 0.4")
        ET.SubElement(table, "geom", 
                     name="table_top",
                     type="box", 
                     size="0.4 0.3 0.02", 
                     rgba="0.8 0.6 0.4 1",
                     pos="0 0 0")
        ET.SubElement(table, "geom", 
                     name="table_leg1",
                     type="box", 
                     size="0.02 0.02 0.2", 
                     rgba="0.6 0.4 0.2 1",
                     pos="0.35 0.25 -0.2")
        ET.SubElement(table, "geom", 
                     name="table_leg2",
                     type="box", 
                     size="0.02 0.02 0.2", 
                     rgba="0.6 0.4 0.2 1",
                     pos="-0.35 0.25 -0.2")
        ET.SubElement(table, "geom", 
                     name="table_leg3",
                     type="box", 
                     size="0.02 0.02 0.2", 
                     rgba="0.6 0.4 0.2 1",
                     pos="0.35 -0.25 -0.2")
        ET.SubElement(table, "geom", 
                     name="table_leg4",
                     type="box", 
                     size="0.02 0.02 0.2", 
                     rgba="0.6 0.4 0.2 1",
                     pos="-0.35 -0.25 -0.2")
        
        # Cube à manipuler (sur la table)
        cube = ET.SubElement(worldbody, "body", name="cube", pos="0.5 0 0.45")
        ET.SubElement(cube, "joint", name="cube_free", type="free")
        ET.SubElement(cube, "geom", 
                     name="cube_geom",
                     type="box", 
                     size="0.025 0.025 0.025", 
                     rgba="0.2 0.8 0.2 1",
                     density="1000",
                     friction="1 0.1 0.1")
        
        # Inclure les modèles G1
        ET.SubElement(root, "include", file="../assets/hands/g1_body.xml")
        ET.SubElement(root, "include", file="../assets/hands/g1_fingers.xml")
        
        # Collecter tous les joints
        joints = []
        for joint in body_tree.findall(".//joint"):
            joint_name = joint.get("name")
            if joint_name:
                joints.append(joint_name)
        
        for joint in fingers_tree.findall(".//joint"):
            joint_name = joint.get("name")
            if joint_name:
                joints.append(joint_name)
        
        # Actuateurs
        actuator = ET.SubElement(root, "actuator")
        for joint_name in joints:
            ET.SubElement(actuator, "position", 
                         name=f"act_{joint_name}", 
                         joint=joint_name, 
                         gear="1",
                         kp="100",
                         kv="10")
        
        # Capteurs
        sensor = ET.SubElement(root, "sensor")
        for joint_name in joints:
            ET.SubElement(sensor, "jointpos", name=f"pos_{joint_name}", joint=joint_name)
            ET.SubElement(sensor, "jointvel", name=f"vel_{joint_name}", joint=joint_name)
        
        # Capteurs pour le cube
        ET.SubElement(sensor, "framepos", name="cube_pos", objtype="body", objname="cube")
        ET.SubElement(sensor, "framequat", name="cube_quat", objtype="body", objname="cube")
        
        # Sauvegarder
        tree = ET.ElementTree(root)
        ET.indent(tree, space="  ", level=0)
        tree.write(self.output_file, encoding="utf-8", xml_declaration=True)
        
        print(f"✅ Modèle créé: {self.output_file}")
        print(f"  - {len(joints)} joints avec actuateurs")
        print(f"  - Environnement: sol, table, cube")
        print(f"  - {len(joints) * 2} capteurs")
    
    def build(self):
        """Construit le modèle complet"""
        print("🚀 CONSTRUCTION DU MODÈLE G1 MANIPULATION")
        print("=" * 60)
        
        try:
            # Vérifications
            self.verify_files()
            
            # Corrections
            self.fix_mesh_paths(self.body_file)
            self.fix_mesh_paths(self.fingers_file)
            self.fix_frictionloss(self.body_file)
            self.fix_frictionloss(self.fingers_file)
            
            # Création
            self.create_combined_model()
            
            print("\n🎉 MODÈLE CRÉÉ AVEC SUCCÈS!")
            print("=" * 60)
            print(f"✅ Fichier: {self.output_file}")
            print("✅ Environnement de manipulation prêt")
            print("✅ Tous les chemins corrigés")
            print("✅ Attributs frictionloss optimisés")
            
            return True
            
        except Exception as e:
            print(f"\n❌ ERREUR: {e}")
            return False

def main():
    """Point d'entrée principal"""
    builder = G1ModelBuilder()
    success = builder.build()
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()
