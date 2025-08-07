#!/usr/bin/env python3
"""
🧪 TEST DU MODÈLE SAC GRASPING
=============================

Script pour tester un modèle entraîné et générer des vidéos
"""

import os
import sys
import argparse
import numpy as np

# Ajouter le workspace au path
sys.path.append('/workspace')

def main():
    """Fonction principale de test"""
    
    parser = argparse.ArgumentParser(description='🧪 Test du modèle SAC')
    parser.add_argument('--model', type=str, default='/workspace/sac_results/models/best_model.zip',
                       help='Chemin vers le modèle')
    parser.add_argument('--episodes', type=int, default=3,
                       help='Nombre d\'épisodes de test')
    parser.add_argument('--video-dir', type=str, default='/workspace/test_videos',
                       help='Dossier de vidéos')
    
    args = parser.parse_args()
    
    print("🧪 TEST DU MODÈLE SAC GRASPING")
    print("=" * 40)
    print(f"🧠 Modèle: {args.model}")
    print(f"🎬 Épisodes: {args.episodes}")
    print(f"📁 Vidéos: {args.video_dir}")
    print("=" * 40)
    
    try:
        from grasp_env import GraspEnv
        from stable_baselines3 import SAC
        
        # Créer le dossier vidéo
        os.makedirs(args.video_dir, exist_ok=True)
        
        # Charger le modèle
        if not os.path.exists(args.model):
            print(f"❌ Modèle non trouvé: {args.model}")
            return 1
        
        print(f"📥 Chargement du modèle...")
        model = SAC.load(args.model)
        print(f"✅ Modèle chargé")
        
        # Créer l'environnement avec vidéo
        env = GraspEnv(render_mode="rgb_array", record_video=True, video_dir=args.video_dir)
        
        # Tester le modèle
        total_rewards = []
        success_count = 0
        
        for episode in range(args.episodes):
            print(f"\n🎬 Épisode {episode + 1}/{args.episodes}")
            
            obs, _ = env.reset()
            total_reward = 0
            steps = 0
            phases_reached = []
            
            for step in range(500):
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = env.step(action)
                total_reward += reward
                steps += 1
                
                # Suivre les phases
                current_phase = info.get('phase', 'UNKNOWN')
                if current_phase not in phases_reached:
                    phases_reached.append(current_phase)
                    print(f"   ✅ Phase atteinte: {current_phase}")
                
                if step % 100 == 0:
                    contact = "✅" if info.get('contact_detected', False) else "❌"
                    grasped = "✅" if info.get('cube_grasped', False) else "❌"
                    lifted = "✅" if info.get('cube_lifted', False) else "❌"
                    print(f"   Step {step:3d} | Contact: {contact} | Saisi: {grasped} | Levé: {lifted}")
                
                if terminated or truncated:
                    break
            
            total_rewards.append(total_reward)
            
            # Analyser le succès
            final_phase = info.get('phase', 'UNKNOWN')
            phases_count = len(phases_reached)
            
            # Succès si au moins GRASP atteint
            success = 'GRASP' in phases_reached or 'LIFT' in phases_reached or 'HOLD' in phases_reached
            if success:
                success_count += 1
            
            print(f"   📊 Résultats:")
            print(f"      🏆 Récompense: {total_reward:.2f}")
            print(f"      📏 Steps: {steps}")
            print(f"      📚 Phases: {' → '.join(phases_reached)}")
            print(f"      🎯 Succès: {'✅' if success else '❌'}")
            
            # Sauvegarder la vidéo
            video_name = f"test_ep{episode + 1:02d}_reward{total_reward:.0f}.mp4"
            env.save_video(video_name)
            print(f"      🎬 Vidéo: {video_name}")
        
        env.close()
        
        # Statistiques finales
        print(f"\n📊 STATISTIQUES FINALES:")
        print(f"   🎬 Épisodes testés: {args.episodes}")
        print(f"   🏆 Récompense moyenne: {np.mean(total_rewards):.2f}")
        print(f"   📈 Récompense max: {np.max(total_rewards):.2f}")
        print(f"   📉 Récompense min: {np.min(total_rewards):.2f}")
        print(f"   🎯 Taux de succès: {success_count}/{args.episodes} ({100*success_count/args.episodes:.1f}%)")
        
        print(f"\n📁 VIDÉOS GÉNÉRÉES:")
        for i, reward in enumerate(total_rewards):
            video_file = f"test_ep{i + 1:02d}_reward{reward:.0f}.mp4"
            video_path = os.path.join(args.video_dir, video_file)
            print(f"   📹 {video_path}")
        
        print(f"\n✅ TEST TERMINÉ!")
        
        return 0
        
    except Exception as e:
        print(f"\n❌ Erreur: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)