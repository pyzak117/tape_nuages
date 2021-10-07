# -*- coding: utf-8 -*-

# Standard libraries
import os
import json

# Data-science libraries
import numpy as  np
np.seterr(divide='ignore', invalid='ignore')

import bottleneck as bn
from matplotlib import pyplot as plt

# Geo libraries
import gdal
from PIL import Image as pillow_img

# Image class
from image import Image

def cloudVerif(zone, imgPath, T1, t2):

    # Importation des images
    image = Image(imgPath, zone)

    # Calcul ci1
    ci1 = image.compute_ci1()
    ci1Abs = abs(ci1-1)

    # Calcul ci2
    ci2 = image.compute_ci2()

    # Classification ci1
    vi1 = image.classif_ci1(ci1Abs, T1)

    # Classification ci2
    vi2 = image.classif_ci2(ci2, t2)

    # Calcul de la fusion
    fusion = image.fusionVisuals(vi1,vi2)

    # Visualisation de la fusion
    # image.array2png(fusion, 1, "fusion.png")

    ratio = image.surface_nuage(fusion)

    return ratio

# Boucle sur toutes les images présentes dans le dossier de la série temporelle

if __name__ == "__main__":

    # Get configuration variables
    with open("conf.json", "r") as fileConf:
        conf = json.load(fileConf)

    # Unpack Conf
    xMin = conf["TOPLEFT"][0]
    yMax = conf["TOPLEFT"][1]    
    xMax = conf["BOTRIGHT"][0]
    yMin = conf["BOTRIGHT"][1]

    zone = [xMin, xMax, yMin, yMax]

    seuilZone = conf["SEUIL_ZONE"]
    pathSerie = conf["PATH"]
    resultFile = conf["RESULT_FILE"]

    T1 = conf["T1"]
    t2 = conf["t2"]

    dic = {}
    
    # Création du fichier de résultats
    with open(resultFile, "w") as rf:
        rf.write("")

    # Appel de la fonction de calcul sur toutes les images
    for folder in os.listdir(pathSerie):
        pathImage = os.path.join(pathSerie, folder)
        dic[folder] = cloudVerif(zone, pathImage, T1, t2)
        print(dic)

    # Ecriture du fichier final avec les résultats
    for ratio in dic:
        if dic[ratio] > seuilZone:
            with open(resultFile, "a") as rf:
                rf.write(ratio + " = " + str(dic[ratio]))