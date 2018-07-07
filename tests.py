from models.base_model import Generator, Discriminator

G = Generator(3, 3)
D = Discriminator(3)

print("Generator", G)
print("Discrimin", D)