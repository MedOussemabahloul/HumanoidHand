from PIL import Image
import os

os.makedirs('assets/textures', exist_ok=True)

# skin.png (couleur peau)
skin = Image.new('RGB', (64, 64), (230, 200, 140))
skin.save('assets/textures/skin.png')

# wood.png (couleur bois clair)
wood = Image.new('RGB', (64, 64), (190, 160, 100))
wood.save('assets/textures/wood.png')

# metal.png (gris clair)
metal = Image.new('RGB', (64, 64), (180, 180, 185))
metal.save('assets/textures/metal.png')

print("Textures créées dans assets/textures/")
