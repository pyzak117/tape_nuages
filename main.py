# -*- coding: utf-8 -*-

"""
~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Auteurs : Pellen Julien, Duvanel Thibaut
Base : https://doi.org/10.1016/j.isprsjprs.2018.07.006
Date : Oct 2021
~~~~~~~~~~~~~~~~~~~~~~~~~~~~
"""

# Standard libraries
import os
import json

# Image class
from image import Image

def main(fileConfPath = "conf.json"):
    
    # Récupération des paramètres contenus dans le fichier conf.json
    with open(fileConfPath, "r") as fileConf:
        conf = json.load(fileConf)

    xMin = conf["TOPLEFT"][0]
    yMax = conf["TOPLEFT"][1]    
    xMax = conf["BOTRIGHT"][0]
    yMin = conf["BOTRIGHT"][1]
    T1 = conf["T1"]
    t2 = conf["t2"]
    seuilZone = conf["SEUIL_ZONE"]
    pathSerie = conf["PATH"]
    resultFile = conf["RESULT_FILE"]

    results = {}

    # Calcul du taux de couverture pour chaque image de la serie
    for nomImage in os.listdir(pathSerie):

        # Reconstitution du chemin de l'image
        imgPath = os.path.join(pathSerie, nomImage)

        # Importation des images
        image = Image(imgPath, [xMin, xMax, yMin, yMax])

        # Calcul des indices nuageux
        ci1 = image.compute_ci1()
        ci1Abs = abs(ci1-1)
        ci2 = image.compute_ci2()

        # Classification à partir des matrices d'indice et des seuils
        vi1 = image.classifIndice(ci1Abs, 1, T1)
        vi2 = image.classifIndice(ci2, 2, t2)

        # Fusion des deux classifications
        fusion = image.fusionClassifs(vi1,vi2)

        # Visualisation de la fusion
        image.matrice2png(fusion)

        txCouv = image.surfaceNonZero(fusion)

        # Ajout du resultat au dictionnaire de résultats
        results[nomImage] = txCouv

    # Ecriture d'un fichier texte à partir du dictionnaire résultats
    for nomImage in results:
        txCouv = results[nomImage]
        # if txCouv < seuilZone:
        with open(resultFile, "a") as rf:
            rf.write(nomImage + " = " + str(txCouv) + "\n")

    print(results)

if __name__ == "__main__":
    main()
