# Simulation de Grasping G1 Robot

Ce projet contient des simulations de grasping pour le robot G1 avec détection de contact et génération de vidéos.

## 🎯 Objectif

Créer des vidéos de simulation de la phase de grasping où le robot G1 :
- Détecte le contact avec le cube via les capteurs de force
- Effectue le grasping en fermant les doigts sur le cube
- Utilise le modèle `g1_combined.xml`
- Implémente des récompenses et pénalités appropriées

## 📁 Fichiers de Simulation

### 1. `grasp_simulation_simple.py` (Recommandé)
- **Version simple** qui fonctionne avec le modèle existant
- Détection de contact basée sur la distance entre les doigts et le cube
- Aucune modification du modèle XML requise
- Génère une vidéo et un fichier de récompenses

### 2. `grasp_simulation.py`
- Version utilisant les capteurs de force existants
- Détection de contact via les capteurs de force
- Fonctionne avec le modèle actuel

### 3. `grasp_simulation_improved.py`
- Version améliorée qui ajoute des capteurs de force au modèle
- Détection de contact plus sophistiquée
- Modifie automatiquement le fichier XML

## 🚀 Utilisation Rapide

### Option 1: Interface Interactive
```bash
python run_grasp_simulation.py
```
Puis suivez le menu pour choisir la simulation souhaitée.

### Option 2: Lancement Direct
```bash
# Test de configuration
python test_grasp_simulation.py

# Simulation simple (recommandée)
python grasp_simulation_simple.py

# Simulation avec force sensors
python grasp_simulation.py

# Simulation améliorée
python grasp_simulation_improved.py
```

## 📋 Prérequis

### Dépendances Python
```bash
pip install mujoco mujoco_viewer numpy opencv-python
```

### Fichiers Requis
- `results/g1_combined.xml` - Modèle du robot G1
- `assets/hands/g1_body.xml` - Corps du robot
- `assets/hands/g1_fingers.xml` - Doigts du robot

## 🎮 Phases de Grasping

La simulation suit 4 phases principales :

1. **Approach** : Approche avec doigts ouverts
2. **Contact** : Détection du contact avec l'objet
3. **Grasp** : Fermeture des doigts pour la préhension
4. **Lift** : Levage de l'objet

## 🏆 Système de Récompenses

### Récompenses Positives
- **Contact** : +10.0 pour détection de contact
- **Grasp Force** : +5.0 × force de préhension
- **Lift Height** : +2.0 × hauteur de levage
- **Stability** : +1.0 pour stabilité du cube
- **Grasp Success** : +50.0 pour succès complet

### Pénalités
- **Energy** : -0.1 × vitesse des doigts (économie d'énergie)

## 📊 Fichiers de Sortie

### Vidéos
- `grasp_simulation_simple.mp4` - Vidéo de la simulation simple
- `grasp_simulation.mp4` - Vidéo de la simulation avec force sensors
- `grasp_simulation_improved.mp4` - Vidéo de la simulation améliorée

### Données
- `grasp_rewards_simple.txt` - Historique des récompenses (simple)
- `grasp_rewards.txt` - Historique des récompenses (force sensors)
- `grasp_rewards_improved.txt` - Historique des récompenses (améliorée)

## 🔧 Configuration

### Paramètres Modifiables
Dans chaque fichier de simulation, vous pouvez ajuster :

```python
# Seuils de détection
contact_threshold = 0.05  # Seuil de détection de contact
grasp_force_threshold = 0.5  # Seuil de force de préhension
lift_height_threshold = 0.06  # Seuil de hauteur de levage

# Poids des récompenses
reward_weights = {
    "contact": 10.0,
    "grasp_force": 5.0,
    "lift_height": 2.0,
    "stability": 1.0,
    "energy_penalty": -0.1,
    "grasp_success": 50.0
}
```

### Positions des Doigts
```python
# Doigts ouverts
open_positions = [0.0] * len(finger_joint_ids)

# Doigts fermés
closed_positions = [1.0] * len(finger_joint_ids)

# Doigts en position de préhension
grasp_positions = [0.6] * len(finger_joint_ids)
```

## 🐛 Dépannage

### Erreur "Modèle non trouvé"
- Vérifiez que `results/g1_combined.xml` existe
- Vérifiez les chemins relatifs dans les fichiers

### Erreur "Capteur non trouvé"
- La version simple ne nécessite pas de capteurs de force
- Utilisez `grasp_simulation_simple.py` pour éviter cette erreur

### Erreur "OpenCV"
- Installez OpenCV : `pip install opencv-python`
- Vérifiez que votre système supporte l'encodage vidéo

### Simulation trop lente
- Réduisez `time.sleep()` dans les boucles
- Diminuez la résolution de capture vidéo
- Utilisez moins de frames pour la vidéo

## 📈 Améliorations Possibles

1. **Détection de Contact Plus Précise**
   - Utiliser les capteurs tactiles existants
   - Implémenter une détection basée sur les collisions

2. **Contrôle Adaptatif**
   - Ajuster la force de préhension en temps réel
   - Optimiser les positions des doigts

3. **Environnements Variés**
   - Ajouter différents objets à saisir
   - Varier les positions initiales

4. **Analyse des Performances**
   - Métriques de succès plus détaillées
   - Comparaison entre différentes stratégies

## 🤝 Contribution

Pour améliorer les simulations :

1. Testez d'abord avec `test_grasp_simulation.py`
2. Modifiez les paramètres dans les fichiers de simulation
3. Documentez vos changements
4. Testez avec différentes configurations

## 📞 Support

En cas de problème :
1. Vérifiez les prérequis
2. Lancez le test de configuration
3. Consultez les messages d'erreur
4. Vérifiez la compatibilité des versions

---

**Note** : Ces simulations sont conçues pour le modèle G1 avec le fichier `g1_combined.xml`. Assurez-vous que votre modèle est compatible avant utilisation.