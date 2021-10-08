# -*- coding: utf-8 -*-

# Standard libraries
import os
import re

# Datascience libraries
import numpy as np

# Geo libraries
import gdal
from PIL import Image as pillow_img

class Image:
    """
    This class represent a satellite image. By default, a Landsat-8 self.
    The __init__ function only need a path to the image folder, and it built numpy arrays for each band.
    """

    def __init__(self, imgPath, zone, sensor = "LS8"):

        bands = {}
        xMin, xMax, yMin, yMax = zone

        if xMin >= xMax or yMin >= yMax :
            raise ValueError("Coordonnées de la zone d'intérêt invalides")

        print("Recherche et chargement des canaux")

        for fichier in os.listdir(imgPath):
            print(fichier)
            
            # Si c'est une image .tif
            if fichier.upper().endswith(".TIF"):

                # Recherche du numéro de la bande dans le nom de l'image
                try:
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

                    # Transformation en matrice numpy sur la zone d'intérêt
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
        ci1 = (self.blue + self.green + self.red)/(self.nir + self.swir1*2)
        return ci1

    def compute_ci2(self):
        ci2 = (self.blue + self.green + self.red + self.nir + self.swir1 + self.swir2) / 6
        return ci2
    
    def classif_ci1(self, matrice, T1 = 0.1):

        # Classification suspects / non suspects
        suspects = []
        for srcLine in matrice:
            trgtLine = []

            for pixel in srcLine:
                if pixel < T1:
                    trgtLine.append(255)
                else:
                    trgtLine.append(0)

            suspects.append(trgtLine)

        npSuspects = np.array(suspects)

        # Dialogue avec user pour savoir comment voir les résultats
        show = input("     >>> Visuel prêt. Affichage ? (Y/N)\n     >>>")

        if show.upper() == "Y":
            pillow_img.fromarray(npSuspects).convert("L").show()
        
        save = input("     >>> Sauvegarde? (Y/N)\n     >>>")
    
        if save.upper() == "Y":

            path = input("Emplacement:\n>>> ")
            path += ".png"

            try:
                pillow_img.fromarray(npSuspects).convert("L").save(path)

            except FileNotFoundError:
                print("Repertoire non existant.")

        return npSuspects

    def classif_ci2(self, ci2, t2 = 0.1):

        # Définition du seuil T2
        meanci2 = np.mean(ci2[~np.isnan(ci2)])
        maxci2 = np.max(ci2[~np.isnan(ci2)])
        T2 = meanci2 + (t2 * (maxci2 - meanci2))

        # Classification suspects / non suspects
        suspects = []

        for srcLine in ci2:
            trgtLine = []

            for pixel in srcLine:
                if pixel > T2:
                    trgtLine.append(255)
                else:
                    trgtLine.append(0)
            
            suspects.append(trgtLine)
        npSuspects = np.array(suspects)

        # Communication utilisateur
        show = input("     >>> Visuel prêt. Affichage ? (Y/N)\n     >>>")

        if show.upper() == "Y":
            pillow_img.fromarray(npSuspects).convert("L").show()
        
        save = input("     >>> Sauvegarde? (Y/N)\n     >>>")
        
        if save.upper() == "Y":

            path = input("Emplacement:\n>>> ")
            path += ".png"

            try:
                pillow_img.fromarray(npSuspects).convert("L").save(path)

            except ValueError:
                print("Nom de fichier invalide.")
        
        return npSuspects

    def fusionVisuals(self, visual1, visual2):
        return visual1 + visual2

        """
        print("Construction de la matrice de visualisation de la fusion des deux indices")
        sizeX = len(visual1)
        sizeY = len(visual1[0])
        fusion = []
        curY = 0

        for y in range(sizeY):
            line = []
            curX = 0

            for x in range(sizeX):
                if visual1[curY][curX] == 255 or visual2[curY][curX] == 255:
                    line.append(255)
                else:
                    line.append(0)
                curX += 1
            
            fusion.append(line)
            curY += 1
        
        print("     >>> Terminé.")
        return np.array(fusion)
        """

    def surface_nuage(self, matrice):
        l_mask =[]
        for line in matrice : 
            for pixel in line : 
                if pixel >0:
                    l_mask.append(pixel)
        nb_pixel_nuage = len(l_mask)
        surface = len(matrice) * len(matrice[0])

        return (nb_pixel_nuage / surface) * 100

    def array2png(self, array, mode = 0, path = ""):
        if mode == 0:
            pillow_img.fromarray(array).convert("L").show()
        
        else:
            print("Ecriture disque de la matrice de fusion")
            
            try:
                pillow_img.fromarray(array).convert("L").save(path)
            except ValueError:
                print("Chemin d'enregistrement de l'image inconnu")