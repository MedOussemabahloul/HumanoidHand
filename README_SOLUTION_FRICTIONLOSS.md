# Solution au Problème Frictionloss dans MuJoCo

## 🔍 Problème Identifié

L'erreur que vous rencontrez :
```
❌ Erreur de chargement: XML Error: Schema violation: unrecognized attribute: 'frictionloss'
Element 'general', line 0
```

**Cause** : L'attribut `frictionloss` n'est supporté que dans certains éléments MuJoCo :
- `<joint>` ✅
- `<tendon>` ✅ 
- `<spatial>` ✅
- `<fixed>` ✅

Il **n'est PAS supporté** dans :
- `<general>` ❌
- `<actuator>` ❌
- Autres éléments ❌

## 🛠️ Solutions Fournies

### 1. Script de Correction Automatique

**Fichier** : `scripts/fix_frictionloss.py`

**Usage** :
```bash
# Corriger le fichier g1_fingers.xml
python scripts/fix_frictionloss.py assets/hands/g1_fingers.xml

# Ou spécifier un fichier de sortie
python scripts/fix_frictionloss.py assets/hands/g1_fingers.xml assets/hands/g1_fingers_fixed.xml
```

**Ce que fait le script** :
- ✅ Supprime `frictionloss` des éléments où il n'est pas autorisé
- ✅ Conserve `frictionloss` dans les joints et tendons
- ✅ Ajoute `frictionloss="0.01"` aux joints qui n'en ont pas
- ✅ Crée une sauvegarde automatique

### 2. Script de Test Corrigé

**Fichier** : `scripts/test_stability_fixed.py`

**Usage** :
```bash
python scripts/test_stability_fixed.py
```

**Fonctionnalités** :
- ✅ Teste le chargement des modèles
- ✅ Vérifie la stabilité de simulation
- ✅ Rapport détaillé des performances

### 3. Générateur de Modèle Combiné Corrigé

**Fichier** : `scripts/build_combine_fixed.py`

**Usage** :
```bash
python scripts/build_combine_fixed.py
```

**Fonctionnalités** :
- ✅ Combine g1_body.xml et g1_fingers.xml
- ✅ Nettoie automatiquement les attributs frictionloss
- ✅ Génère des actuateurs optimisés
- ✅ Ajoute des capteurs de surveillance

## 🚀 Procédure de Correction Recommandée

### Étape 1 : Corriger les Fichiers Existants
```bash
# Corriger g1_fingers.xml
python scripts/fix_frictionloss.py assets/hands/g1_fingers.xml

# Vérifier que g1_body.xml n'a pas le même problème
python scripts/fix_frictionloss.py assets/hands/g1_body.xml
```

### Étape 2 : Tester les Corrections
```bash
# Tester tous les modèles
python scripts/test_stability_fixed.py
```

### Étape 3 : Créer le Modèle Combiné Optimisé
```bash
# Générer le modèle combiné corrigé
python scripts/build_combine_fixed.py
```

### Étape 4 : Vérification Finale
```bash
# Tester le nouveau modèle combiné
python -c "
import mujoco
model = mujoco.MjModel.from_xml_path('results/g1_combined_fixed.xml')
print('✅ Modèle chargé avec succès!')
print(f'Joints: {model.njnt}, Actuateurs: {model.nu}')
"
```

## 📋 Référence Technique

### Attributs Frictionloss Autorisés

**Dans les joints** :
```xml
<joint name="mon_joint" type="hinge" frictionloss="0.01"/>
```

**Dans les tendons spatiaux** :
```xml
<spatial name="mon_tendon" frictionloss="0.02">
    <site site="site1"/>
    <site site="site2"/>
</spatial>
```

**Dans les tendons fixes** :
```xml
<fixed name="mon_tendon_fixe" frictionloss="0.01">
    <joint joint="joint1" coef="1"/>
</fixed>
```

### Attributs Frictionloss NON Autorisés

**❌ Dans les actuateurs** :
```xml
<!-- INCORRECT -->
<general name="actuator1" frictionloss="0.01"/>
<motor name="motor1" frictionloss="0.01"/>
<position name="pos1" frictionloss="0.01"/>
```

## 🔧 Valeurs Recommandées

| Élément | Valeur Frictionloss | Usage |
|---------|-------------------|-------|
| Joint rotatif | 0.01 - 0.05 | Friction dans les roulements |
| Joint prismatique | 0.02 - 0.1 | Friction de glissement |
| Tendon spatial | 0.01 - 0.03 | Friction dans les poulies |
| Tendon fixe | 0.005 - 0.02 | Friction interne |

## 🆘 Dépannage

### Si le script de correction échoue :
1. Vérifiez que le fichier XML est valide
2. Assurez-vous d'avoir les permissions d'écriture
3. Vérifiez la syntaxe XML avec un validateur

### Si les tests de stabilité échouent :
1. Réduisez le timestep dans les options
2. Augmentez les paramètres de solver
3. Vérifiez les limites des joints

### Si le modèle combiné ne se charge pas :
1. Vérifiez les chemins des fichiers inclus
2. Assurez-vous que tous les assets sont accessibles
3. Validez la syntaxe XML du fichier généré

## 📚 Documentation de Référence

- [MuJoCo XML Reference](https://mujoco.readthedocs.io/en/stable/XMLreference.html)
- [Joint Documentation](https://mujoco.readthedocs.io/en/stable/XMLreference.html#body-joint)
- [Tendon Documentation](https://mujoco.readthedocs.io/en/stable/XMLreference.html#tendon)
- [Actuator Documentation](https://mujoco.readthedge.io/en/stable/XMLreference.html#actuator)

## ✅ Validation de la Solution

Après avoir appliqué les corrections, vous devriez voir :
```
🔍 Test de chargement: results/g1_combined_fixed.xml
✅ Modèle chargé avec succès!
   - Nombre de corps: XX
   - Nombre de joints: XX
   - Nombre de degrés de liberté: XX
   - Nombre d'actuateurs: XX
```

Au lieu de :
```
❌ Erreur de chargement: XML Error: Schema violation: unrecognized attribute: 'frictionloss'
```