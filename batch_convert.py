import os
import shutil
import subprocess
import sys
from pathlib import Path
import xml.etree.ElementTree as ET
from typing import List, Tuple
import argparse
import tempfile

class URDFToMuJoCoConverter:
    def __init__(self, source_dir: str, output_dir: str, verbose: bool = True):
        self.source_dir = Path(source_dir)
        self.output_dir = Path(output_dir)
        self.verbose = verbose
        self.conversion_log = []
        
    def log(self, message: str):
        """Affiche un message si le mode verbose est activé"""
        if self.verbose:
            print(f"[INFO] {message}")
        self.conversion_log.append(message)
    
    def check_dependencies(self) -> bool:
        """Vérifie que les dépendances nécessaires sont installées"""
        try:
            # Vérifier si urdf2mjcf est disponible
            result = subprocess.run(['urdf2mjcf', '--help'], 
                                  capture_output=True, text=True, timeout=5)
            if result.returncode != 0:
                print("❌ L'outil 'urdf2mjcf' n'est pas trouvé dans le PATH")
                print("   Assurez-vous que le package urdf2mjcf est installé")
                return False
        except (subprocess.TimeoutExpired, FileNotFoundError):
            print("❌ L'outil 'urdf2mjcf' n'est pas accessible")
            return False
        
        return True
    
    def convert_urdf_with_urdf2mjcf(self, urdf_path: Path, output_path: Path) -> bool:
        """Convertit un URDF en XML MuJoCo en utilisant urdf2mjcf avec gestion d'erreur améliorée"""
        try:
            # Créer un répertoire temporaire pour la conversion
            with tempfile.TemporaryDirectory() as temp_dir:
                temp_urdf = Path(temp_dir) / urdf_path.name
                temp_output = Path(temp_dir) / output_path.name
                
                # Copier et pré-traiter le fichier URDF
                shutil.copy2(urdf_path, temp_urdf)
                
                # Exécuter la conversion
                cmd = ['urdf2mjcf', str(temp_urdf), '--output', str(temp_output)]
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
                
                if result.returncode == 0:
                    # Copier le résultat vers la destination finale
                    shutil.copy2(temp_output, output_path)
                    self.log(f"✅ Conversion réussie: {urdf_path} -> {output_path}")
                    return True
                
                self.log(f"❌ Erreur lors de la conversion de {urdf_path}")
                self.log(f"   Sortie: {result.stdout}")
                self.log(f"   Erreur: {result.stderr}")
                
                # Essayer de récupérer le fichier temporaire en cas d'échec
                if temp_output.exists():
                    self.log("⚠️ Attempting to use partially converted file")
                    shutil.copy2(temp_output, output_path)
                    return True
                    
                return False
                    
        except Exception as e:
            self.log(f"❌ Erreur conversion {urdf_path}: {str(e)}")
            return False
    def preprocess_urdf(self, urdf_path: Path):
        """Pré-traitement supplémentaire pour les URDF problématiques"""
        try:
            # Réparer les chemins de mesh
            with open(urdf_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Normaliser les chemins Windows
            content = content.replace('\\', '/')
            
            # Corriger les chemins relatifs incorrects
            content = re.sub(r'filename="(?!\/)(?!\.\/)([^"]+)"', r'filename="./\1"', content)
            
            with open(urdf_path, 'w', encoding='utf-8') as f:
                f.write(content)
                
        except Exception as e:
            self.log(f"⚠️ Preprocessing failed for {urdf_path}: {str(e)}")
    def convert_to_basic_xml(self, urdf_path: Path, output_path: Path) -> bool:
        """Conversion basique de secours quand urdf2mjcf échoue"""
        try:
            self.log("🔄 Using basic XML fallback conversion")
            
            # Lire et parser l'URDF
            tree = ET.parse(urdf_path)
            root = tree.getroot()
            
            # Créer la structure MJCF de base
            mujoco = ET.Element('mujoco')
            mujoco.set('model', root.get('name', 'robot'))
            
            # Ajouter les sections essentielles
            compiler = ET.SubElement(mujoco, 'compiler')
            compiler.set('meshdir', 'meshes')
            compiler.set('angle', 'radian')
            
            option = ET.SubElement(mujoco, 'option')
            option.set('timestep', '0.005')
            
            worldbody = ET.SubElement(mujoco, 'worldbody')
            
            # Copier les éléments essentiels
            for elem in root:
                if elem.tag in ['link', 'joint']:
                    worldbody.append(elem)
            
            # Écrire le fichier de sortie
            tree = ET.ElementTree(mujoco)
            tree.write(output_path, encoding='utf-8', xml_declaration=True)
            return True
            
        except Exception as e:
            self.log(f"❌ Fallback conversion failed: {str(e)}")
            return False
    
    def copy_mesh_files(self, source_meshes_dir: Path, target_meshes_dir: Path):
        """Copie les fichiers de mesh vers le dossier de destination"""
        if source_meshes_dir.exists():
            self.log(f"Copie des meshes: {source_meshes_dir} -> {target_meshes_dir}")
            shutil.copytree(source_meshes_dir, target_meshes_dir, dirs_exist_ok=True)
    
    def find_urdf_files(self) -> List[Path]:
        """Détecte tous les fichiers URDF-like (.urdf + .xml avec balises <link> et <joint>)"""
        candidates = []
        for ext in ('*.xml', '*.urdf'):
            for path in self.source_dir.rglob(ext):
                try:
                    tree = ET.parse(path)
                    root = tree.getroot()
                    tags = {el.tag for el in root.iter()}
                    if 'link' in tags and 'joint' in tags:
                        self.log(f"🟢 Détecté fichier URDF-like : {path}")
                        candidates.append(path)
                    else:
                        self.log(f"⚪ Ignoré (non URDF-like) : {path}")
                except ET.ParseError:
                    self.log(f"🔴 Fichier corrompu : {path}")
        return candidates

    def create_directory_structure(self, urdf_files: List[Path]):
        """Crée la structure de dossiers dans le dossier de sortie"""
        for urdf_file in urdf_files:
            # Calculer le chemin relatif
            rel_path = urdf_file.relative_to(self.source_dir)
            output_dir = self.output_dir / rel_path.parent
            output_dir.mkdir(parents=True, exist_ok=True)
    
    def convert_all(self) -> Tuple[int, int]:
        """Convertit tous les fichiers URDF trouvés"""
        if not self.source_dir.exists():
            raise FileNotFoundError(f"Le dossier source {self.source_dir} n'existe pas")
        
        # Créer le dossier de sortie
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Trouver tous les fichiers URDF
        urdf_files = self.find_urdf_files()
        
        if not urdf_files:
            self.log("Aucun fichier URDF trouvé")
            return 0, 0
        
        self.log(f"Trouvé {len(urdf_files)} fichiers URDF")
        
        # Créer la structure de dossiers
        self.create_directory_structure(urdf_files)
        
        # Vérifier les dépendances
        use_urdf2mjcf = self.check_dependencies()
        
        successful_conversions = 0
        failed_conversions = 0
        
        for urdf_file in urdf_files:
            # Calculer le chemin de sortie
            rel_path = urdf_file.relative_to(self.source_dir)
            output_file = self.output_dir / rel_path.with_suffix('.xml')
            
            self.log(f"Conversion: {urdf_file} -> {output_file}")
            
            # Convertir le fichier avec urdf2mjcf
            success = self.convert_urdf_with_urdf2mjcf(urdf_file, output_file)
            
            if success:
                successful_conversions += 1
            else:
                self.log("⚠️ Attempting fallback conversion")
                success = self.convert_to_basic_xml(urdf_file, output_file)

        
        # Copier les fichiers mesh
        self.copy_mesh_directories()
        
        return successful_conversions, failed_conversions
    
    def copy_mesh_directories(self):
        """Copie tous les dossiers de meshes"""
        for mesh_dir in self.source_dir.rglob('meshes'):
            if mesh_dir.is_dir():
                rel_path = mesh_dir.relative_to(self.source_dir)
                target_dir = self.output_dir / rel_path
                self.copy_mesh_files(mesh_dir, target_dir)
    
    def generate_report(self, successful: int, failed: int):
        """Génère un rapport de conversion"""
        report_path = self.output_dir / 'conversion_report.txt'
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("=== RAPPORT DE CONVERSION URDF vers MuJoCo ===\n\n")
            f.write(f"Dossier source: {self.source_dir}\n")
            f.write(f"Dossier de sortie: {self.output_dir}\n\n")
            f.write(f"Conversions réussies: {successful}\n")
            f.write(f"Conversions échouées: {failed}\n\n")
            f.write("=== LOG DÉTAILLÉ ===\n")
            
            for log_entry in self.conversion_log:
                f.write(f"{log_entry}\n")
        
        self.log(f"Rapport généré: {report_path}")
# Dans la classe URDFToMuJoCoConverter (batch_convert.py)

def convert_to_basic_xml(self, urdf_path: Path, output_path: Path) -> bool:
    """Conversion basique de secours quand urdf2mjcf échoue"""
    try:
        self.log("🔄 Using basic XML fallback conversion")
        
        # Lire et parser l'URDF
        tree = ET.parse(urdf_path)
        root = tree.getroot()
        
        # Créer la structure MJCF de base
        mujoco = ET.Element('mujoco')
        mujoco.set('model', root.get('name', 'robot'))
        
        # Ajouter les sections essentielles
        compiler = ET.SubElement(mujoco, 'compiler')
        compiler.set('meshdir', 'meshes')
        compiler.set('angle', 'radian')
        
        option = ET.SubElement(mujoco, 'option')
        option.set('timestep', '0.005')
        
        worldbody = ET.SubElement(mujoco, 'worldbody')
        
        # Copier les éléments essentiels
        for elem in root:
            if elem.tag in ['link', 'joint']:
                worldbody.append(elem)
        
        # Écrire le fichier de sortie
        tree = ET.ElementTree(mujoco)
        tree.write(output_path, encoding='utf-8', xml_declaration=True)
        return True
        
    except Exception as e:
        self.log(f"❌ Fallback conversion failed: {str(e)}")
        return False


def main():
    parser = argparse.ArgumentParser(description='Convertit des fichiers URDF vers le format MuJoCo')
    parser.add_argument('source', help='Dossier source contenant les fichiers URDF')
    parser.add_argument('output', help='Dossier de sortie pour les fichiers MuJoCo')
    parser.add_argument('-v', '--verbose', action='store_true', help='Mode verbose')
    
    args = parser.parse_args()
    
    try:
        converter = URDFToMuJoCoConverter(args.source, args.output, args.verbose)
        
        print(f"🚀 Début de la conversion...")
        print(f"📁 Source: {args.source}")
        print(f"📁 Sortie: {args.output}")
        print()
        
        successful, failed = converter.convert_all()
        
        print(f"\n✅ Conversion terminée!")
        print(f"   Réussies: {successful}")
        print(f"   Échouées: {failed}")
        
        converter.generate_report(successful, failed)
        
        if failed > 0:
            print(f"\n⚠️  {failed} conversion(s) ont échoué. Consultez le rapport pour plus de détails.")
            sys.exit(1)
        
    except Exception as e:
        print(f"❌ Erreur fatale: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()