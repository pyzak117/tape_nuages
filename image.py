# -*- coding: utf-8 -*-

# Standard libraries
import os
import re

# Third-party libraries
import numpy as np
import gdal
from PIL import Image as pillow_img

class Image:
    """
    Cette classe représente une image satellite multispectrale. 
    Pour le moment, seul les produits Landsat-8 sont pris en charge.

    attributs
    ---------
        blue(ndarray) : contient les valeurs de la bande bleue de l'image
        red(ndarray) : contient les valeurs de la bande rouge de l'image
        ...

    """

    def __init__(self, imgPath, zone, sensor = "LS8"):

        bands = {}
        xMin, xMax, yMin, yMax = zone

        if xMin >= xMax or yMin >= yMax :
            raise ValueError("Coordonnées de la zone d'intérêt invalides")

        print("Recherche et chargement des canaux")

        # Parcours de tous les fichiers dans le dossier de l'image
        for fichier in os.listdir(imgPath):
            
            # Si c'est une image .tif
            if fichier.upper().endswith(".TIF"):

                try:
                    # Recherche du numéro de la bande dans le nom de l'image
                    motif = re.compile("B[0-9]+.TIF$")
                    pos = re.search(motif, fichier.upper()).span()[0]

                    # Construction de l'identifiant de la bande
                    bandId = fichier[pos:-4]

                    # Construction du chemin complet du fichier
                    fileBandName = os.path.join(imgPath, fichier)

                    # Chargement de l'image dans un dataset
                    ds = gdal.Open(fileBandName)

                    # Récupération d'un geotransform qui contient données géographiques
                    geoTransform = ds.GetGeoTransform()

                    # Unpack du geotransform
                    orX = geoTransform[0]
                    orY = geoTransform[3]
                    largeurPixel = geoTransform[1]
                    hauteurPixel = geoTransform[5]

                    # Cadrage de la zone
                    row1=int((yMax-orY)/hauteurPixel)
                    col1=int((xMin-orX)/largeurPixel)
                    row2=int((yMin-orY)/hauteurPixel)
                    col2=int((xMax-orX)/largeurPixel)

                    # Transformation en matrice numpy uniquement sur la zone d'intérêt
                    ar = ds.ReadAsArray(col1,row1,col2-col1+1,row2-row1+1).astype(np.float32)

                    # Stockage de l'array dans le dictionnaire bands
                    bands[bandId] = ar

                    print(">>> Canal {} chargé".format(bandId))

                except AttributeError:
                    print(">>> Fichier {} ignoré".format(fichier))

        if sensor == "LS8":
            self.blue = bands["B2"]
            self.green = bands["B3"]
            self.red = bands["B4"]
            self.nir = bands["B5"]
            self.swir1 = bands["B6"]
            self.swir2 = bands["B7"]

    def compute_ci1(self):
        """
        Calcule l'indice nuageux ci1
        """
        ci1 = (self.blue + self.green + self.red)/(self.nir + self.swir1*2)
        return ci1

    def compute_ci2(self):
        """
        Calcule l'indice nuageux ci2
        """
        ci2 = (self.blue + self.green + self.red + self.nir + self.swir1 + self.swir2) / 6
        return ci2

    def classifIndice(self, matriceIndice, mode, T1 = 0.1, t2 = 0.1):
        
        # Si on souhaite produire une classification basée sur une matrice de résultats de l'indice T1
        if mode == 1:
            classif = np.where(matriceIndice < T1, 255, 0)

        # Si on souhaite produire une classification basée sur une matrice de résultats de l'indice T2
        elif mode == 2:

            # Définition du seuil T2 en fonction du paramètre t2
            meanci2 = np.mean(matriceIndice[~np.isnan(matriceIndice)])
            maxci2 = np.max(matriceIndice[~np.isnan(matriceIndice)])
            T2 = meanci2 + (t2 * (maxci2 - meanci2))

            # Calcul de la classification
            classif = np.where(matriceIndice > T2, 255, 0)

            # Classification pixels qui respectent la condition VS ceux qui ne la respectent pas

        else:
            raise ValueError("Mode inconnu")

        return classif

    def fusionClassifs(self, visual1, visual2):
        return visual1 + visual2

    def surface_nuage(self, matrice):
        return (np.count_nonzero(matrice)/(np.shape(matrice)[0]*np.shape(matrice)[1]))*100

    def matrice2png(self, matrice, mode = 0, path = ""):
        if mode == 0:
            pillow_img.fromarray(matrice).convert("L").show()
        
        elif mode == 1:
            print("Ecriture disque de la matrice de fusion")
            
            try:
                pillow_img.fromarray(matrice).convert("L").save(path)

            except ValueError:
                print("Chemin d'enregistrement de l'image inconnu")
