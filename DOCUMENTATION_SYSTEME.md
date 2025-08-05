# Documentation du Système de Recherche et Grasping G1

## 🎯 Vue d'ensemble

Ce système simule un robot G1 qui recherche activement un cube avec ses mains, puis effectue un grasping une fois le contact détecté. Le système comprend deux phases distinctes :

1. **Phase de recherche** : Le robot bouge ses bras pour chercher le cube
2. **Phase de grasping** : Une fois le contact détecté, le robot ferme ses doigts

## 🤖 Actuators du Robot G1

### **Actuators de Mouvement des Bras (Recherche)**

Les actuators suivants sont conçus pour le mouvement des bras pendant la phase de recherche :

```python
# Actuators de mouvement des bras
self.arm_movement_actuators = {
    'left_shoulder_pitch': None,   # Épaule gauche - mouvement avant/arrière
    'left_elbow': None,            # Coude gauche - flexion/extension
    'left_wrist': None,            # Poignet gauche - rotation
    'right_shoulder_pitch': None,  # Épaule droite - mouvement avant/arrière
    'right_elbow': None,           # Coude droit - flexion/extension
    'right_wrist': None,           # Poignet droit - rotation
}
```

**Fonction de chaque actuator :**
- **Shoulder Pitch** : Contrôle le mouvement avant/arrière des bras
- **Elbow** : Contrôle la flexion/extension des coudes
- **Wrist** : Contrôle la rotation des poignets

### **Actuators des Doigts (Grasping)**

Les actuators des doigts sont utilisés pour la fermeture des mains :

```python
# Actuators des doigts (16 au total)
self.finger_actuator_ids = [
    # Main gauche (8 actuators)
    'act_left_thumb_joint_0',    # Pouce gauche - articulation 1
    'act_left_thumb_joint_1',    # Pouce gauche - articulation 2
    'act_left_index_joint_0',    # Index gauche - articulation 1
    'act_left_index_joint_1',    # Index gauche - articulation 2
    'act_left_middle_joint_0',   # Majeur gauche - articulation 1
    'act_left_middle_joint_1',   # Majeur gauche - articulation 2
    'act_left_ring_joint_0',     # Annulaire gauche - articulation 1
    'act_left_ring_joint_1',     # Annulaire gauche - articulation 2
    
    # Main droite (8 actuators)
    'act_right_thumb_joint_0',   # Pouce droit - articulation 1
    'act_right_thumb_joint_1',   # Pouce droit - articulation 2
    'act_right_index_joint_0',   # Index droit - articulation 1
    'act_right_index_joint_1',   # Index droit - articulation 2
    'act_right_middle_joint_0',  # Majeur droit - articulation 1
    'act_right_middle_joint_1',  # Majeur droit - articulation 2
    'act_right_ring_joint_0',    # Annulaire droit - articulation 1
    'act_right_ring_joint_1',    # Annulaire droit - articulation 2
]
```

## 🔍 Patterns de Recherche

Le robot utilise des patterns de mouvement prédéfinis pour rechercher le cube :

```python
self.search_patterns = [
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

**Explication des patterns :**
- Chaque pattern définit des positions cibles pour les épaules gauche et droite
- Les valeurs représentent [pitch, roll, yaw] en radians
- Le robot alterne entre ces patterns pour couvrir une zone de recherche

## 🎁 Système de Récompenses

### **Récompenses de la Phase de Recherche**

```python
# Récompense pour être proche du cube
if min_distance < self.search_radius:
    reward += 5.0 * (1.0 - min_distance / self.search_radius)

# Pénalité pour être trop loin
if min_distance > 0.3:
    reward -= 1.0

# Récompense pour le mouvement (encourager l'exploration)
reward += 0.1
```

**Explication :**
- **Récompense de proximité** : Plus le robot est proche du cube, plus il reçoit de points
- **Pénalité de distance** : Le robot est pénalisé s'il s'éloigne trop du cube
- **Récompense de mouvement** : Encourage l'exploration active

### **Récompenses de Contact et Grasping**

```python
# Récompense pour le contact
if self.contact_detected:
    reward += 10.0

# Récompenses pour le grasping
if not self.search_phase and self.grasp_completed:
    reward += 20.0
    
    # Récompense basée sur la stabilité du cube
    if self.grasp_stable:
        reward += 30.0
    
    # Récompense basée sur la hauteur du cube
    cube_height = cube_pos[2]
    reward += cube_height * 10.0
```

**Explication :**
- **Contact** : +10 points pour détecter le contact avec le cube
- **Grasping complet** : +20 points pour fermer complètement les doigts
- **Stabilité** : +30 points si le cube est stable dans la main
- **Hauteur** : +10 × hauteur pour encourager le soulèvement du cube

### **Pénalités**

```python
# Pénalité pour les mouvements excessifs du cube
if distance > 0.15:
    reward -= distance * 20.0

# Pénalité pour le temps (encourager l'efficacité)
if self.step_count > 1000:
    reward -= 0.2
```

**Explication :**
- **Mouvements excessifs** : Pénalise si le cube s'éloigne trop de sa position initiale
- **Pénalité temporelle** : Encourage à accomplir la tâche rapidement

## 🔧 Paramètres Configurables

### **Paramètres de Recherche**

```python
self.contact_threshold = 0.05    # Seuil de détection de contact
self.search_radius = 0.2         # Rayon de recherche autour du cube
self.search_speed = 0.1          # Vitesse de mouvement de recherche
self.pattern_duration = 100      # Steps par pattern de mouvement
```

### **Paramètres de Grasping**

```python
self.open_position = 0.0         # Position ouverte des doigts
self.closed_position = 1.5       # Position fermée des doigts
self.grasp_force = 1.5           # Force de fermeture des doigts
```

## 📊 Métriques de Performance

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

## 🎮 Utilisation du Système

### **Lancement d'une Simulation**

```bash
# Démo rapide avec 3 configurations
python run_search_grasp_demo.py --demo

# Simulation personnalisée
python run_search_grasp_demo.py --steps 2000 --search-radius 0.2 --grasp-force 1.5
```

### **Paramètres de Commande**

- `--steps` : Nombre maximum d'étapes
- `--contact-threshold` : Seuil de détection de contact
- `--search-radius` : Rayon de recherche autour du cube
- `--search-speed` : Vitesse de mouvement de recherche
- `--grasp-force` : Force de fermeture des doigts

## 🔍 Détection de Contact

### **Sensors Utilisés**

```python
# Force sensors (simulés via les sensors de position des doigts)
self.force_sensor_ids = [
    # 32 sensors de position des doigts
    'pos_left_thumb_joint_0', 'pos_left_thumb_joint_1',
    'pos_left_index_joint_0', 'pos_left_index_joint_1',
    # ... etc
]

# Contact sensors (optionnels)
self.contact_sensor_ids = [
    # Sensors de contact spécifiques (si disponibles)
]
```

### **Logique de Détection**

```python
def _detect_contact(self):
    # Lire les valeurs des force sensors
    force_values = [abs(self.data.sensordata[i]) for i in self.force_sensor_ids]
    
    # Détecter le contact si une valeur dépasse le seuil
    max_force = max(force_values)
    return max_force > self.contact_threshold
```

## 📈 Améliorations Possibles

1. **Force sensors réels** : Intégrer de vrais force sensors
2. **Contact sensors** : Ajouter des contact sensors spécifiques
3. **Apprentissage** : Intégrer un système d'apprentissage par renforcement
4. **Visualisation** : Ajouter une interface graphique
5. **Multi-objets** : Supporter plusieurs objets à saisir

---

**Système de Recherche et Grasping G1** - Documentation complète ! 🚀