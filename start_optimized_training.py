#!/usr/bin/env python3
"""
🚀 DÉMARRAGE TRAINING OPTIMISÉ
===============================

Script simple pour lancer le training inspiré du collègue
avec nos améliorations professionnelles.
"""

import os
import sys

def main():
    """Lance le training optimisé"""
    
    print("🚀 DÉMARRAGE DU TRAINING OPTIMISÉ")
    print("=" * 50)
    print("📋 Configuration:")
    print("   ✅ Inspiré du collègue (scaling adaptatif, reset contrôles)")
    print("   ✅ Notre valeur ajoutée (curriculum, mouvements fluides)")
    print("   ✅ XML corrigé (plus d'erreurs NaN/inf)")
    print("   ✅ TD3 avec hyperparamètres qui fonctionnent")
    print("=" * 50)
    
    # Configuration headless comme le collègue
    os.environ["MUJOCO_GL"] = "egl"
    
    try:
        # Import et lancement
        from optimized_training import main as training_main
        training_main()
        
    except KeyboardInterrupt:
        print("\n⏹️ Training interrompu par l'utilisateur")
    except ImportError as e:
        print(f"❌ Erreur d'import: {e}")
        print("💡 Assurez-vous que toutes les dépendances sont installées:")
        print("   pip3 install --break-system-packages numpy mujoco gymnasium stable-baselines3 imageio pillow")
    except Exception as e:
        print(f"❌ Erreur: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()