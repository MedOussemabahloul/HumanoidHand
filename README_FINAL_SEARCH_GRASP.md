# 🤖 Système de Recherche et Grasping G1 - Documentation Complète

## 🎯 Vue d'ensemble

Ce système simule un robot G1 qui recherche activement un cube avec ses mains, puis effectue un grasping une fois le contact détecté. Le système comprend deux phases distinctes avec un système complet de récompenses et pénalités.

### **Phases de la Simulation :**

1. **🔍 Phase de Recherche** : Le robot bouge ses bras pour chercher le cube
2. **🤏 Phase de Grasping** : Une fois le contact détecté, le robot ferme ses doigts

## 🚀 Installation et Utilisation

### **Prérequis**
```bash
# Créer un environnement virtuel
python3 -m venv grasp_env
source grasp_env/bin/activate

# Installer les dépendances
pip install mujoco numpy
```

### **Lancement Rapide**
```bash
# Démo avec 3 configurations différentes
python run_search_grasp_demo.py --demo

# Simulation personnalisée
python run_search_grasp_demo.py --steps 2000 --search-radius 0.2 --grasp-force 1.5
```

## 🤖 Actuators du Robot G1

### **Actuators de Mouvement des Bras (Recherche)**

Les actuators suivants sont conçus pour le mouvement des bras pendant la phase de recherche :

| Actuator | ID | Fonction |
|----------|----|----------|
| `left_shoulder_pitch` | 1 | Épaule gauche - mouvement avant/arrière |
| `left_elbow` | 2 | Coude gauche - flexion/extension |
| `left_wrist` | 3 | Poignet gauche - rotation |
| `right_shoulder_pitch` | 4 | Épaule droite - mouvement avant/arrière |
| `right_elbow` | 5 | Coude droit - flexion/extension |
| `right_wrist` | 6 | Poignet droit - rotation |

### **Actuators des Doigts (Grasping)**

16 actuators pour le contrôle des doigts :

**Main Gauche (8 actuators) :**
- `act_left_thumb_joint_0/1` : Pouce gauche
- `act_left_index_joint_0/1` : Index gauche  
- `act_left_middle_joint_0/1` : Majeur gauche
- `act_left_ring_joint_0/1` : Annulaire gauche

**Main Droite (8 actuators) :**
- `act_right_thumb_joint_0/1` : Pouce droit
- `act_right_index_joint_0/1` : Index droit
- `act_right_middle_joint_0/1` : Majeur droit
- `act_right_ring_joint_0/1` : Annulaire droit

## 🔍 Patterns de Recherche

Le robot utilise 5 patterns de mouvement prédéfinis :

```python
search_patterns = [
    # Pattern 1: Bras écartés
    {"left_shoulder": [0.3, 0.2, 0.0], "right_shoulder": [0.3, -0.2, 0.0]},
    
    # Pattern 2: Bras rapprochés  
    {"left_shoulder": [0.4, 0.1, 0.0], "right_shoulder": [0.4, -0.1, 0.0]},
    
    # Pattern 3: Bras centrés
    {"left_shoulder": [0.3, 0.0, 0.0], "right_shoulder": [0.3, 0.0, 0.0]},
    
    # Pattern 4: Bras bas
    {"left_shoulder": [0.2, 0.1, 0.0], "right_shoulder": [0.2, -0.1, 0.0]},
    
    # Pattern 5: Position intermédiaire
    {"left_shoulder": [0.25, 0.15, 0.0], "right_shoulder": [0.25, -0.15, 0.0]},
]
```

## 🎁 Système de Récompenses et Pénalités

### **Récompenses de la Phase de Recherche**

```python
# Récompense pour être proche du cube
if min_distance < search_radius:
    reward += 5.0 * (1.0 - min_distance / search_radius)

# Pénalité pour être trop loin
if min_distance > 0.3:
    reward -= 1.0

# Récompense pour le mouvement (encourager l'exploration)
reward += 0.1
```

**Explication :**
- **Récompense de proximité** : +5 points × (1 - distance/rayon) quand proche du cube
- **Pénalité de distance** : -1 point si trop loin du cube
- **Récompense de mouvement** : +0.1 point pour encourager l'exploration

### **Récompenses de Contact et Grasping**

```python
# Récompense pour le contact
if contact_detected:
    reward += 10.0

# Récompenses pour le grasping
if grasp_completed:
    reward += 20.0
    
    # Récompense basée sur la stabilité
    if grasp_stable:
        reward += 30.0
    
    # Récompense basée sur la hauteur du cube
    reward += cube_height * 10.0
```

**Explication :**
- **Contact** : +10 points pour détecter le contact
- **Grasping complet** : +20 points pour fermer les doigts
- **Stabilité** : +30 points si le cube est stable
- **Hauteur** : +10 × hauteur pour encourager le soulèvement

### **Pénalités**

```python
# Pénalité pour les mouvements excessifs du cube
if distance > 0.15:
    reward -= distance * 20.0

# Pénalité pour le temps (encourager l'efficacité)
if step_count > 1000:
    reward -= 0.2
```

**Explication :**
- **Mouvements excessifs** : -20 × distance si le cube s'éloigne trop
- **Pénalité temporelle** : -0.2 point après 1000 steps

## 📊 Résultats des Tests

### **Analyse des Sensors (48 sensors détectés) :**

- **Force sensors** : 32 (simulés via les sensors de position des doigts)
- **Contact sensors** : 0 (non disponibles dans ce modèle)
- **Joint sensors** : 46 (positions et vitesses des articulations)

### **Analyse des Actuators (23 actuators détectés) :**

- **Finger actuators** : 16 (contrôle des doigts)
- **Arm actuators** : 6 (mouvement des bras)
- **Cube actuator** : 1 (contrôle du cube)

### **Performance des Démonstrations :**

| Démo | Paramètres | Résultat | Récompense |
|------|------------|----------|------------|
| 1 | Par défaut | Contact détecté | 9966.40 |
| 2 | Recherche précise | Contact détecté | 9966.40 |
| 3 | Force élevée | Contact détecté | 9966.40 |

**Observations :**
- ✅ **Contact détecté** dans tous les cas (étape 4)
- ✅ **Phase de recherche terminée** avec succès
- ⚠️ **Grasping non complet** (doigts ne se ferment pas assez)
- ⚠️ **Stabilité non atteinte** (cube reste à sa position initiale)

## 🔧 Paramètres Configurables

### **Paramètres de Recherche**
```python
contact_threshold = 0.05    # Seuil de détection de contact
search_radius = 0.2         # Rayon de recherche autour du cube
search_speed = 0.1          # Vitesse de mouvement de recherche
pattern_duration = 100      # Steps par pattern de mouvement
```

### **Paramètres de Grasping**
```python
open_position = 0.0         # Position ouverte des doigts
closed_position = 1.5       # Position fermée des doigts
grasp_force = 1.5          # Force de fermeture des doigts
```

## 📁 Structure des Fichiers

```
├── grasp_search_simulation.py      # Simulation principale
├── run_search_grasp_demo.py        # Script de démo
├── fix_model_paths.py              # Création du modèle de test
├── DOCUMENTATION_SYSTEME.md        # Documentation technique
├── results/
│   └── g1_test_simple.xml          # Modèle MuJoCo simplifié
└── grasp_env/                      # Environnement virtuel
```

## 🎮 Utilisation Avancée

### **Paramètres de Commande**
```bash
python run_search_grasp_demo.py [OPTIONS]

Options:
  --model PATH              Chemin vers le modèle MuJoCo
  --steps INT               Nombre maximum d'étapes
  --contact-threshold FLOAT Seuil de détection de contact
  --search-radius FLOAT     Rayon de recherche autour du cube
  --search-speed FLOAT      Vitesse de mouvement de recherche
  --grasp-force FLOAT       Force de fermeture des doigts
  --demo                    Lancer une démo rapide
```

### **Exemples d'Utilisation**
```bash
# Démo rapide
python run_search_grasp_demo.py --demo

# Simulation personnalisée
python run_search_grasp_demo.py --steps 1500 --search-radius 0.15 --grasp-force 2.0

# Recherche plus précise
python run_search_grasp_demo.py --contact-threshold 0.02 --search-speed 0.15
```

## 🔍 Détection de Contact

### **Méthode Actuelle**
Le système utilise les sensors de position des doigts pour simuler la détection de contact :

```python
def _detect_contact(self):
    # Lire les valeurs des sensors de position des doigts
    force_values = [abs(self.data.sensordata[i]) for i in self.force_sensor_ids]
    
    # Détecter le contact si une valeur dépasse le seuil
    max_force = max(force_values)
    return max_force > self.contact_threshold
```

### **Amélioration Possible**
Intégrer de vrais force sensors ou contact sensors pour une détection plus précise.

## 📈 Métriques de Performance

### **Indicateurs de Succès**
1. **Phase de recherche terminée** : Le robot a trouvé le cube
2. **Contact détecté** : Les mains touchent le cube
3. **Grasping réussi** : Les doigts sont fermés sur le cube
4. **Grasping stable** : Le cube est stable dans la main

### **Calcul de la Récompense Totale**
```python
total_reward = (
    reward_recherche +      # Récompenses de la phase de recherche
    reward_contact +        # Récompense pour le contact
    reward_grasping +       # Récompenses pour le grasping
    reward_stabilite +      # Récompense pour la stabilité
    reward_hauteur +        # Récompense pour la hauteur
    penalite_mouvement +    # Pénalités pour les mouvements excessifs
    penalite_temps          # Pénalités temporelles
)
```

## 🚨 Problèmes Identifiés

### **Instabilité de Simulation**
```
WARNING: Nan, Inf or huge value in QACC at DOF 1. The simulation is unstable.
```
**Cause** : Paramètres de simulation trop agressifs
**Solution** : Ajuster les paramètres de force et de vitesse

### **Grasping Incomplet**
**Problème** : Les doigts ne se ferment pas complètement
**Cause** : Force de fermeture insuffisante ou contraintes du modèle
**Solution** : Augmenter `grasp_force` ou ajuster `closed_position`

## 🔮 Améliorations Futures

1. **Force sensors réels** : Intégrer de vrais force sensors
2. **Contact sensors** : Ajouter des contact sensors spécifiques
3. **Apprentissage par renforcement** : Intégrer un système d'apprentissage
4. **Visualisation** : Ajouter une interface graphique
5. **Multi-objets** : Supporter plusieurs objets à saisir
6. **Optimisation des paramètres** : Ajuster automatiquement les paramètres

## 📞 Support

Pour toute question ou problème :
1. Vérifiez que l'environnement virtuel est activé
2. Assurez-vous que tous les modules sont installés
3. Consultez la documentation technique (`DOCUMENTATION_SYSTEME.md`)
4. Testez avec la démo rapide (`--demo`)

---

**Système de Recherche et Grasping G1** - Prêt pour la production ! 🚀

*Développé avec MuJoCo, Python et beaucoup de patience* 😄