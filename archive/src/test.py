import sys
import os

print(sys.path)
print("\n")

root_path = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
print(root_path)
sys.path.append(root_path)
print("\n")

print(sys.path)
# from utils.eaml.dataloader import EAML_DataLoader