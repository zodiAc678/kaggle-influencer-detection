# main.py
from nbclient import NotebookClient
from nbformat import read
from extract_features import *


# Fonction à utiliser pour l'extraction des features
function = extract_user_features
stop_steps = 2000



# Fonction pour choisir quelle extraction utiliser
def extract_data_from_features(row):
    # Example function that uses the feature extraction functions
    return function(row)

def extract_method():
    return function.__name__

nb = read("bertweet_with_features.ipynb", as_version=4)
client = NotebookClient(nb)

print("Début du notebook...")
try:
    client.execute()
except SystemExit:
    print("Fin du notebook suite à SystemExit.")