
#!/usr/bin/env python3
"""
🔧 DIAGNOSTIC ET RÉPARATION SEGFAULT
==================================

Script pour diagnostiquer et réparer les segmentation faults
Causes communes: dépendances manquantes, versions incompatibles, problèmes MuJoCo
"""

import sys
import os
import subprocess
import traceback
from pathlib import Path

def run_safe_command(cmd, description):
    """Exécuter commande en sécurité"""
    print(f"🔧 {description}...")
    try:
        result = subprocess.run(cmd, shell=True, check=True, 
                               capture_output=True, text=True, timeout=60)
        print(f"   ✅ {description}: OK")
        return True, result.stdout
    except subprocess.TimeoutExpired:
        print(f"   ⏰ {description}: TIMEOUT")
        return False, "Timeout"
    except subprocess.CalledProcessError as e:
        print(f"   ❌ {description}: ERREUR")
        print(f"   📜 Stderr: {e.stderr[:200]}...")
        return False, e.stderr
    except Exception as e:
        print(f"   💥 {description}: EXCEPTION - {e}")
        return False, str(e)

def check_python_environment():
    """Vérifier environnement Python"""
    print("🐍 DIAGNOSTIC ENVIRONNEMENT PYTHON")
    
    # Version Python
    print(f"   Python: {sys.version}")
    
    # Virtual env
    venv = os.environ.get('VIRTUAL_ENV')
    if venv:
        print(f"   ✅ Virtual env: {venv}")
    else:
        print("   ⚠️  Pas de virtual env détecté")
    
    # Modules critiques test sécurisé
    critical_modules = ['sys', 'os', 'subprocess', 'pathlib']
    for module in critical_modules:
        try:
            __import__(module)
            print(f"   ✅ {module}: OK")
        except ImportError:
            print(f"   ❌ {module}: MANQUANT (critique!)")
            return False
    
    return True

def diagnose_import_issues():
    """Diagnostiquer problèmes d'import spécifiques"""
    print("\n📦 DIAGNOSTIC IMPORTS CRITIQUES")
    
    # Test imports un par un avec capture d'erreur
    modules_to_test = {
        'numpy': 'import numpy as np',
        'torch': 'import torch',
        'mujoco': 'import mujoco',
        'yaml': 'import yaml',
        'scipy': 'import scipy'
    }
    
    failed_imports = []
    
    for module_name, import_cmd in modules_to_test.items():
        try:
            # Test import dans subprocess séparé pour éviter crash
            test_script = f"""
import sys
try:
    {import_cmd}
    print(f"SUCCESS: {module_name}")
except ImportError as e:
    print(f"IMPORT_ERROR: {module_name} - {{e}}")
except Exception as e:
    print(f"OTHER_ERROR: {module_name} - {{e}}")
"""
            
            result = subprocess.run([sys.executable, '-c', test_script], 
                                  capture_output=True, text=True, timeout=10)
            
            if "SUCCESS" in result.stdout:
                print(f"   ✅ {module_name}: Import OK")
            elif "IMPORT_ERROR" in result.stdout:
                print(f"   ❌ {module_name}: Import Failed")
                failed_imports.append(module_name)
            else:
                print(f"   💥 {module_name}: Crash lors import")
                failed_imports.append(module_name)
                
        except subprocess.TimeoutExpired:
            print(f"   ⏰ {module_name}: Timeout (possible segfault)")
            failed_imports.append(module_name)
        except Exception as e:
            print(f"   💥 {module_name}: Exception - {e}")
            failed_imports.append(module_name)
    
    return failed_imports

def test_mujoco_specifically():
    """Test spécifique MuJoCo (cause fréquente segfault)"""
    print("\n🔬 DIAGNOSTIC MUJOCO SPÉCIFIQUE")
    
    # Test 1: Import MuJoCo
    test_script = """
import sys
try:
    import mujoco
    print("MUJOCO_IMPORT: SUCCESS")
    print(f"MUJOCO_VERSION: {mujoco.__version__}")
    
    # Test basique
    print("TESTING_BASIC_OPERATIONS")
    
    # Test XML simple
    xml_content = '''
    <mujoco>
        <worldbody>
            <body name="test">
                <geom name="test_geom" type="box" size="0.1 0.1 0.1"/>
            </body>
        </worldbody>
    </mujoco>
    '''
    
    model = mujoco.MjModel.from_xml_string(xml_content)
    data = mujoco.MjData(model)
    print("MUJOCO_TEST: SUCCESS")
    
except ImportError as e:
    print(f"MUJOCO_IMPORT: FAILED - {e}")
except Exception as e:
    print(f"MUJOCO_ERROR: {e}")
"""
    
    try:
        result = subprocess.run([sys.executable, '-c', test_script], 
                              capture_output=True, text=True, timeout=15)
        
        if "MUJOCO_IMPORT: SUCCESS" in result.stdout:
            print("   ✅ MuJoCo import: OK")
            if "MUJOCO_TEST: SUCCESS" in result.stdout:
                print("   ✅ MuJoCo test basique: OK")
                return True
            else:
                print("   ❌ MuJoCo test basique: ÉCHEC")
                print(f"   📜 Output: {result.stdout}")
                return False
        else:
            print("   ❌ MuJoCo import: ÉCHEC")
            print(f"   📜 Stderr: {result.stderr}")
            return False
            
    except subprocess.TimeoutExpired:
        print("   💥 MuJoCo: TIMEOUT (probable segfault)")
        return False
    except Exception as e:
        print(f"   💥 MuJoCo: Exception - {e}")
        return False

def test_script_syntax():
    """Tester syntaxe du script sans l'exécuter"""
    print("\n📝 DIAGNOSTIC SYNTAXE SCRIPT")
    
    script_path = "scripts/train_sac_per_ultra.py"
    
    if not os.path.exists(script_path):
        print(f"   ❌ Script non trouvé: {script_path}")
        return False
    
    # Test compilation syntaxe
    try:
        with open(script_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        compile(content, script_path, 'exec')
        print(f"   ✅ Syntaxe script: OK")
        return True
        
    except SyntaxError as e:
        print(f"   ❌ Erreur syntaxe: ligne {e.lineno}")
        print(f"   📜 {e.text}")
        return False
    except Exception as e:
        print(f"   ❌ Erreur compilation: {e}")
        return False

def suggest_fixes(failed_imports, mujoco_ok):
    """Suggérer corrections basées sur diagnostic"""
    print("\n🔧 SUGGESTIONS RÉPARATION")
    
    if failed_imports:
        print("   📦 INSTALLER DÉPENDANCES MANQUANTES:")
        
        # Installation par groupes
        if 'torch' in failed_imports:
            print("   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu")
        
        if 'mujoco' in failed_imports:
            print("   pip install mujoco")
        
        basic_packages = [pkg for pkg in failed_imports if pkg in ['numpy', 'scipy', 'yaml']]
        if basic_packages:
            packages_str = ' '.join(basic_packages)
            if 'yaml' in basic_packages:
                packages_str = packages_str.replace('yaml', 'PyYAML')
            print(f"   pip install {packages_str}")
    
    if not mujoco_ok:
        print("\n   🔬 RÉPARATION MUJOCO:")
        print("   # Réinstallation complète")
        print("   pip uninstall mujoco -y")
        print("   pip install mujoco")
        print("   # Test isolation")
        print("   python -c 'import mujoco; print(mujoco.__version__)'")
    
    print("\n   🚀 COMMANDES SÉCURISÉES APRÈS INSTALLATION:")
    print("   # Vérification système")
    print("   python3 check_requirements.py")
    print("   # Test isolation")
    print("   python3 -c 'import torch, mujoco, numpy; print(\"OK\")'")
    print("   # Entraînement sécurisé")
    print("   python3 launch_training.py --config cpu --debug")

def install_dependencies_safe():
    """Installation sécurisée des dépendances"""
    print("\n📦 INSTALLATION AUTOMATIQUE SÉCURISÉE")
    
    # Installation par étapes pour identifier problèmes
    packages = [
        ("numpy scipy", "Packages scientifiques de base"),
        ("PyYAML", "Configuration YAML"),
        ("torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu", "PyTorch CPU"),
        ("mujoco", "MuJoCo physics engine")
    ]
    
    for package_cmd, description in packages:
        success, output = run_safe_command(f"pip install {package_cmd}", f"Installation {description}")
        if not success:
            print(f"   💥 Échec installation {description}")
            print(f"   💡 Essayez manuellement: pip install {package_cmd}")
            return False
    
    return True

def main():
    """Diagnostic complet et réparation"""
    print("🔧 DIAGNOSTIC ET RÉPARATION SEGMENTATION FAULT")
    print("="*60)
    
    # 1. Environnement Python
    if not check_python_environment():
        print("💥 ERREUR CRITIQUE: Environnement Python corrompu")
        return 1
    
    # 2. Test imports
    failed_imports = diagnose_import_issues()
    
    # 3. Test MuJoCo spécifique
    mujoco_ok = test_mujoco_specifically()
    
    # 4. Test syntaxe script
    syntax_ok = test_script_syntax()
    
    # 5. Résumé diagnostic
    print("\n📊 RÉSUMÉ DIAGNOSTIC")
    print(f"   Imports échoués: {len(failed_imports)}")
    print(f"   MuJoCo OK: {'✅' if mujoco_ok else '❌'}")
    print(f"   Syntaxe OK: {'✅' if syntax_ok else '❌'}")
    
    # 6. Proposer réparation
    if failed_imports or not mujoco_ok:
        print("\n🔧 RÉPARATION AUTOMATIQUE")
        response = input("Voulez-vous installer automatiquement les dépendances ? (y/N): ")
        
        if response.lower() in ['y', 'yes', 'oui']:
            if install_dependencies_safe():
                print("\n✅ INSTALLATION TERMINÉE")
                print("🔄 Relancez le diagnostic: python3 fix_segfault.py")
                print("🚀 Puis testez: python3 launch_training.py --config cpu --debug")
            else:
                suggest_fixes(failed_imports, mujoco_ok)
        else:
            suggest_fixes(failed_imports, mujoco_ok)
    else:
        print("\n✅ DIAGNOSTIC: Dépendances OK")
        print("💡 Le segfault peut venir d'autre chose")
        print("🔍 Essayez: python3 launch_training.py --config cpu --debug")
    
    return 0

if __name__ == "__main__":
    exit(main())
