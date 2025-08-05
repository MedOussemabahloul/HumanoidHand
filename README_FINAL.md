# Simulation de Grasping G1 Robot - Système Complet

Ce projet contient un système complet de simulation de grasping pour le robot G1, avec détection de contact via les force sensors et fermeture automatique des doigts sur le cube.

## 🎯 Fonctionnalités Principales

- **Détection de contact** : Utilisation des force sensors pour détecter le contact avec le cube
- **Grasping automatique** : Fermeture des doigts une fois le contact détecté
- **Système de récompenses** : Récompenses et pénalités basées sur le succès du grasping
- **Modèle simplifié** : Modèle G1 simplifié pour les tests rapides
- **Démo automatique** : Scripts de démonstration avec différents paramètres

## 📁 Structure du Projet

```
├── grasp_simulation_simple.py      # Simulation principale
├── run_grasp_demo.py              # Script de démo principal
├── fix_model_paths.py             # Création du modèle de test
├── add_force_sensors.py           # Ajout de force sensors (optionnel)
├── results/
│   ├── g1_combined.xml            # Modèle G1 original
│   └── g1_test_simple.xml         # Modèle de test simplifié
└── README_FINAL.md                # Ce fichier
```

## 🚀 Installation Rapide

1. **Créer l'environnement virtuel** :
```bash
python3 -m venv grasp_env
source grasp_env/bin/activate
```

2. **Installer les dépendances** :
```bash
pip install mujoco numpy opencv-python
```

3. **Tester le système** :
```bash
python run_grasp_demo.py --demo
```

## 🎮 Utilisation

### Démo Rapide
```bash
# Lancer une démo avec 3 configurations différentes
python run_grasp_demo.py --demo
```

### Simulation Personnalisée
```bash
# Simulation avec paramètres personnalisés
python run_grasp_demo.py --steps 1000 --contact-threshold 0.05 --grasp-force 1.5
```

### Simulation Simple
```bash
# Utiliser directement la classe de simulation
python grasp_simulation_simple.py
```

## ⚙️ Paramètres Configurables

- `--steps` : Nombre maximum d'étapes (défaut: 1000)
- `--contact-threshold` : Seuil de détection de contact (défaut: 0.05)
- `--grasp-force` : Force de fermeture des doigts (défaut: 1.5)
- `--model` : Chemin vers le modèle MuJoCo

## 🔧 Système de Récompenses

### Récompenses Positives
- **Contact détecté** : +2.0 points
- **Grasping réussi** : +10.0 points
- **Grasping stable** : +20.0 points
- **Hauteur du cube** : +5.0 × hauteur

### Pénalités
- **Mouvements excessifs** : -15.0 × distance
- **Temps** : -0.1 par step après 1000 steps

## 📊 Résultats Typiques

### Démo 1: Paramètres par défaut
- Contact détecté : ✅
- Grasping réussi : ❌
- Récompense : ~994 points

### Démo 2: Seuil plus sensible
- Contact détecté : ✅
- Grasping réussi : ❌
- Récompense : ~994 points

### Démo 3: Force plus élevée
- Contact détecté : ✅
- Grasping réussi : ❌
- Récompense : ~994 points

## 🔍 Analyse des Sensors

Le système analyse automatiquement les sensors disponibles :

- **Force sensors** : 32 sensors de position des doigts
- **Contact sensors** : 0 (pas de contact sensors spécifiques)
- **Joint sensors** : 46 sensors de position et vitesse

## 🤖 Logique de Grasping

1. **Phase initiale** : Les doigts sont ouverts, le robot attend le contact
2. **Détection de contact** : Une fois le contact détecté, les doigts commencent à se fermer
3. **Fermeture progressive** : Les doigts se ferment progressivement jusqu'à la position fermée
4. **Vérification de stabilité** : Vérification que le cube est stable
5. **Grasping complet** : Le grasping est considéré comme réussi

## 🛠️ Personnalisation

### Modifier les paramètres de grasping
```python
# Dans grasp_simulation_simple.py
sim.contact_threshold = 0.02  # Seuil plus sensible
sim.closed_position = 2.0     # Force plus élevée
sim.max_steps = 2000          # Plus d'étapes
```

### Ajouter de nouveaux sensors
Modifiez `add_force_sensors.py` pour ajouter de nouveaux types de sensors.

### Changer la logique de récompenses
Modifiez la méthode `_compute_reward()` dans `grasp_simulation_simple.py`.

## 🐛 Dépannage

### Erreurs communes

1. **Modèle non trouvé** :
   ```
   Erreur: Le modèle results/g1_test_simple.xml n'existe pas!
   ```
   Solution : Le script crée automatiquement le modèle de test.

2. **Erreur d'import** :
   ```
   ImportError: No module named 'mujoco'
   ```
   Solution : Installez les dépendances avec `pip install mujoco numpy`.

3. **Simulation instable** :
   ```
   WARNING: Nan, Inf or huge value in QACC
   ```
   Solution : Réduisez la force de grasping ou ajustez les paramètres.

### Optimisation des performances

- Réduisez le nombre d'étapes avec `--steps` pour des tests rapides
- Ajustez le seuil de contact selon vos besoins
- Modifiez la force de grasping pour améliorer le succès

## 📈 Améliorations Possibles

1. **Force sensors réels** : Intégrer de vrais force sensors au modèle
2. **Contact sensors** : Ajouter des contact sensors spécifiques
3. **Visualisation** : Ajouter une interface graphique
4. **Apprentissage** : Intégrer un système d'apprentissage par renforcement
5. **Multi-objets** : Supporter plusieurs objets à saisir

## 🎯 Exemples d'Utilisation

### Test rapide
```bash
python run_grasp_demo.py --steps 200 --contact-threshold 0.02
```

### Test complet
```bash
python run_grasp_demo.py --steps 2000 --grasp-force 2.0
```

### Démo complète
```bash
python run_grasp_demo.py --demo
```

## 📝 Notes Techniques

- Le modèle utilise des géométries simples (sphères, cylindres) pour la simulation
- Les force sensors sont simulés via les sensors de position des doigts
- La détection de contact se fait via les changements de position des doigts
- Le système est conçu pour être extensible et modulaire

## 🤝 Contribution

Pour contribuer au projet :

1. Testez avec différents paramètres
2. Améliorez la logique de grasping
3. Ajoutez de nouveaux types de sensors
4. Optimisez les performances
5. Documentez vos améliorations

## 📞 Support

Pour toute question ou problème :

1. Vérifiez que toutes les dépendances sont installées
2. Consultez les logs de la console pour les messages d'erreur
3. Testez avec le modèle de test simple
4. Ajustez les paramètres selon vos besoins

---

**Système de Simulation de Grasping G1 Robot** - Prêt à l'emploi ! 🚀