# 🤖 G1 Fingers Manipulation - Guide Professionnel

## 📋 Vue d'Ensemble

**Projet :** Simulation de manipulation d'objets avec les mains G1  
**Environnement :** `/home/oussema/Documents/project`  
**Objectif :** Tester le grasping et lifting d'un cube sur une table  

## 📁 Structure Requise

```
/home/oussema/Documents/project/
├── assets/
│   └── hands/
│       ├── g1_body.xml      ✅ Votre fichier
│       ├── g1_fingers.xml   ✅ Votre fichier  
│       └── meshes/          ✅ Vos fichiers .STL
├── results/                 📁 Créé automatiquement
└── scripts/                 📁 Scripts fournis
    ├── create_combined_model.py  ✅ Script 1
    └── test_g1_manipulation.py   ✅ Script 2
```

## 🚀 Utilisation (2 Étapes Seulement)

### Étape 1: Créer le Modèle Combiné

```bash
cd /home/oussema/Documents/project
python scripts/create_combined_model.py
```

**Ce que fait ce script :**
- ✅ Vérifie que vos fichiers existent
- ✅ Corrige automatiquement les chemins des meshes  
- ✅ Corrige les attributs `frictionloss`
- ✅ Crée l'environnement (sol, table, cube)
- ✅ Génère `results/g1_combined.xml`

**Résultat attendu :**
```
🎉 MODÈLE CRÉÉ AVEC SUCCÈS!
✅ Fichier: /home/oussema/Documents/project/results/g1_combined.xml
✅ Environnement de manipulation prêt
✅ Tous les chemins corrigés
✅ Attributs frictionloss optimisés
```

### Étape 2: Tester la Simulation

```bash
python scripts/test_g1_manipulation.py
```

**Ce que fait ce script :**
- ✅ Charge le modèle MuJoCo
- ✅ Lance la simulation interactive 3D
- ✅ Exécute la séquence de manipulation :
  1. **Approche** (3s) : Positionner les mains
  2. **Saisie** (2s) : Fermer les doigts sur le cube
  3. **Levage** (3s) : Soulever le cube
  4. **Maintien** (12s) : Tenir le cube avec oscillations

**Résultat attendu :**
- Fenêtre 3D MuJoCo s'ouvre
- Robot G1 avec mains fonctionnelles
- Table avec cube vert
- Séquence de manipulation automatique

## 🎮 Contrôles de la Simulation

- **ESC** : Quitter la simulation
- **Souris** : Faire tourner la caméra
- **Molette** : Zoom avant/arrière
- **Clic droit + glisser** : Déplacer la vue

## 🔧 Caractéristiques Techniques

### Environnement
- **Sol** : Plan texturé avec grille
- **Table** : 80cm × 60cm × 40cm de hauteur
- **Cube** : 5cm × 5cm × 5cm, vert, sur la table
- **Éclairage** : Directionnel optimisé

### Simulation
- **Fréquence** : 500 Hz (timestep=0.002s)
- **Solveur** : Newton avec tolérance 1e-10
- **Actuateurs** : Position avec gains PD optimisés
- **Capteurs** : Position/vitesse des joints + pose du cube

### Séquence de Manipulation
1. **Phase Approche** : Bras en position d'approche, doigts ouverts
2. **Phase Saisie** : Fermeture progressive des doigts
3. **Phase Levage** : Élévation coordonnée des deux bras
4. **Phase Maintien** : Oscillations légères pour démontrer le contrôle

## ⚠️ Prérequis

```bash
# Installation MuJoCo (si pas déjà fait)
pip install mujoco

# Vérification de l'installation
python -c "import mujoco; print(f'MuJoCo {mujoco.__version__} installé')"
```

## 🔍 Dépannage

### Problème : "Fichier manquant"
```bash
# Vérifiez la structure
ls -la assets/hands/
ls -la assets/hands/meshes/
```

### Problème : "Erreur de chargement MuJoCo"
```bash
# Réexécutez la création du modèle
python scripts/create_combined_model.py
```

### Problème : "Chemins incorrects"
- Les scripts corrigent automatiquement tous les chemins
- Les sauvegardes sont créées (`.backup`)

## 📊 Informations Techniques

| Composant | Détail |
|-----------|--------|
| **Joints** | ~30 (corps + doigts) |
| **Actuateurs** | Position avec gains PD |
| **Capteurs** | Position/vitesse + pose cube |
| **Physique** | Newton solver, contacts optimisés |
| **Rendu** | OpenGL avec textures |

## 🎯 Objectifs Atteints

- ✅ **Modèle combiné** : Corps + doigts unifiés
- ✅ **Environnement réaliste** : Table + cube manipulable  
- ✅ **Séquence automatique** : Approche → Saisie → Levage → Maintien
- ✅ **Simulation interactive** : Visualisation 3D en temps réel
- ✅ **Code professionnel** : Robuste, documenté, sans erreurs

---

**🎉 Votre environnement de manipulation G1 est prêt !**

*Exécutez les 2 scripts dans l'ordre et profitez de la simulation 3D.*