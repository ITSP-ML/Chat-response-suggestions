import os
from dotenv import dotenv_values

path = dotenv_values().get("ITSP_ML_PROJECT_ROOT_PATH")
os.chdir(path)
print(os.getcwd())
