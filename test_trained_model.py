#!/usr/bin/env python3
"""
🧪 TEST DU MODÈLE SAC ENTRAÎNÉ
==============================

Script pour tester et visualiser les performances du robot entraîné.
Génère des vidéos de démonstration des capacités de grasping apprises.

UTILISATION:
python3 test_trained_model.py [--model path/to/model.zip] [--episodes 3]
"""

import os
import sys
import argparse
import time

# Ajouter workspace au path
sys.path.append('/workspace')

def main():
    """Test du modèle entraîné"""
    
    print("🧪 TEST DU MODÈLE SAC ENTRAÎNÉ")
    print("=" * 40)
    print("🤖 Robot G1 - Démonstration Grasping")
    print("🎬 Génération de vidéos de test")
    print("=" * 40)
    
    parser = argparse.ArgumentParser(description='🧪 Test du modèle SAC')
    parser.add_argument('--model', type=str, 
                       default='/workspace/final_results/models/best_model.zip',
                       help='Chemin vers le modèle SAC')
    parser.add_argument('--episodes', type=int, default=3,
                       help='Nombre d\'épisodes de test')
    parser.add_argument('--output-dir', type=str, default='/workspace/test_videos',
                       help='Dossier de sortie des vidéos')
    
    args = parser.parse_args()
    
    print(f"📊 Configuration:")
    print(f"   🧠 Modèle: {args.model}")
    print(f"   📺 Épisodes: {args.episodes}")
    print(f"   📁 Sortie: {args.output_dir}")
    print("=" * 40)
    
    try:
        # Imports
        from stable_baselines3 import SAC
        from robust_grasp_env import RobustGraspEnv
        
        # Vérifier que le modèle existe
        if not os.path.exists(args.model):
            print(f"❌ Modèle introuvable: {args.model}")
            print("💡 Lancez d'abord l'entraînement avec: python3 train_final.py --quick")
            return 1
        
        # Créer le dossier de sortie
        os.makedirs(args.output_dir, exist_ok=True)
        
        # Charger le modèle
        print(f"🔄 Chargement du modèle...")
        model = SAC.load(args.model)
        print(f"✅ Modèle chargé avec succès")
        
        # Créer l'environnement de test
        print(f"🏗️  Création de l'environnement de test...")
        env = RobustGraspEnv(
            render_mode="rgb_array", 
            record_video=True, 
            video_dir=args.output_dir
        )
        print(f"✅ Environnement créé")
        
        # Statistiques globales
        total_rewards = []
        success_count = 0
        phase_stats = {
            'SEARCH': 0, 'APPROACH': 0, 'CONTACT': 0, 
            'ALIGN': 0, 'GRASP': 0, 'LIFT': 0, 'HOLD': 0
        }
        
        print(f"\n🚀 DÉBUT DES TESTS")
        print(f"   🎯 Le robot va démontrer ses capacités de grasping")
        print()
        
        # Exécuter les épisodes de test
        for episode in range(args.episodes):
            print(f"🎬 Épisode {episode + 1}/{args.episodes}")
            
            obs, _ = env.reset()
            total_reward = 0
            steps = 0
            max_phase_reached = 0
            episode_phases = []
            
            start_time = time.time()
            
            for step in range(500):  # Max 500 steps
                # Prédiction déterministe pour consistance
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = env.step(action)
                
                total_reward += reward
                steps += 1
                
                # Collecter statistiques de phase
                phase = info.get('phase', 'UNKNOWN')
                episode_phases.append(phase)
                
                phase_num = ['SEARCH', 'APPROACH', 'CONTACT', 'ALIGN', 'GRASP', 'LIFT', 'HOLD'].index(phase) if phase in ['SEARCH', 'APPROACH', 'CONTACT', 'ALIGN', 'GRASP', 'LIFT', 'HOLD'] else -1
                if phase_num > max_phase_reached:
                    max_phase_reached = phase_num
                
                # Affichage périodique
                if step % 100 == 0:
                    cube_pos = info.get('cube_position', [0, 0, 0])
                    contacts = info.get('finger_contacts', 0)
                    grasped = info.get('cube_grasped', False)
                    lifted = info.get('cube_lifted', False)
                    
                    print(f"   Step {step:3d} | Phase: {phase:8s} | "
                          f"Reward: {reward:6.2f} | Contacts: {contacts} | "
                          f"Saisi: {'✅' if grasped else '❌'} | "
                          f"Levé: {'✅' if lifted else '❌'}")
                
                if terminated or truncated:
                    break
            
            episode_time = time.time() - start_time
            
            # Analyser le succès
            final_phase = episode_phases[-1] if episode_phases else 'UNKNOWN'
            cube_lifted = info.get('cube_lifted', False)
            cube_grasped = info.get('cube_grasped', False)
            
            success = cube_lifted and max_phase_reached >= 5  # Atteint au moins LIFT
            if success:
                success_count += 1
            
            total_rewards.append(total_reward)
            
            # Compter les phases atteintes
            for phase in episode_phases:
                if phase in phase_stats:
                    phase_stats[phase] += 1
            
            print(f"   ✅ Épisode terminé en {episode_time:.1f}s")
            print(f"      🏆 Récompense totale: {total_reward:.2f}")
            print(f"      📈 Steps: {steps}")
            print(f"      🎯 Phase finale: {final_phase}")
            print(f"      🤝 Cube saisi: {'✅' if cube_grasped else '❌'}")
            print(f"      ⬆️  Cube levé: {'✅' if cube_lifted else '❌'}")
            print(f"      🎊 Succès: {'✅' if success else '❌'}")
            
            # Sauvegarder la vidéo
            video_name = f"test_episode_{episode + 1:02d}_reward_{total_reward:.0f}.mp4"
            env.save_video(video_name)
            print(f"      🎬 Vidéo: {video_name}")
            print()
        
        env.close()
        
        # Statistiques finales
        mean_reward = sum(total_rewards) / len(total_rewards)
        max_reward = max(total_rewards)
        min_reward = min(total_rewards)
        success_rate = (success_count / args.episodes) * 100
        
        print(f"📊 STATISTIQUES FINALES")
        print(f"=" * 40)
        print(f"🎯 Épisodes testés: {args.episodes}")
        print(f"🏆 Récompense moyenne: {mean_reward:.2f}")
        print(f"📈 Récompense maximale: {max_reward:.2f}")
        print(f"📉 Récompense minimale: {min_reward:.2f}")
        print(f"🎊 Taux de succès: {success_rate:.1f}%")
        print()
        
        print(f"📈 PROGRESSION PAR PHASE:")
        total_phase_count = sum(phase_stats.values())
        for phase, count in phase_stats.items():
            percentage = (count / total_phase_count) * 100 if total_phase_count > 0 else 0
            print(f"   {phase:8s}: {percentage:5.1f}% ({count:3d} occurrences)")
        print()
        
        print(f"🎬 VIDÉOS GÉNÉRÉES:")
        for i in range(args.episodes):
            video_file = f"test_episode_{i + 1:02d}_reward_{total_rewards[i]:.0f}.mp4"
            video_path = os.path.join(args.output_dir, video_file)
            if os.path.exists(video_path):
                size_mb = os.path.getsize(video_path) / (1024 * 1024)
                print(f"   📹 {video_file} ({size_mb:.1f} MB)")
            else:
                print(f"   ❌ {video_file} (non créé)")
        print()
        
        print(f"🎯 ÉVALUATION DU ROBOT:")
        if success_rate >= 80:
            print(f"   🌟 EXCELLENT - Le robot maîtrise le grasping!")
        elif success_rate >= 60:
            print(f"   ✅ TRÈS BIEN - Bonnes performances de grasping")
        elif success_rate >= 40:
            print(f"   👍 BIEN - Le robot apprend le grasping")
        elif success_rate >= 20:
            print(f"   📈 MOYEN - Progrès visible, nécessite plus d'entraînement")
        else:
            print(f"   📚 DÉBUTANT - Le robot commence à apprendre")
        
        if mean_reward > 5000:
            print(f"   🏆 Récompenses excellentes - Comportement optimal")
        elif mean_reward > 3000:
            print(f"   🎯 Récompenses bonnes - Comportement efficace")
        elif mean_reward > 1000:
            print(f"   📈 Récompenses correctes - Apprentissage en cours")
        else:
            print(f"   📚 Récompenses faibles - Besoin de plus d'entraînement")
        
        print(f"\n🎉 TESTS TERMINÉS AVEC SUCCÈS!")
        print(f"📁 Vidéos disponibles dans: {args.output_dir}")
        
        return 0
        
    except FileNotFoundError as e:
        print(f"❌ Fichier non trouvé: {e}")
        return 1
    except Exception as e:
        print(f"❌ Erreur: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)