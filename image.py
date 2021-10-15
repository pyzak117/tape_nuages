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
import re

# Third-party libraries
import numpy as np
import gdal
from PIL import Image as pillow_img

class Image:
    """
    Cette classe représente une image Landsat-8 multispectrale. 
    Ses attributs sont des matrices ndarray représentant chaque bande
    et sont nommés selon la couleur de la bande qu'ils contiennent.

    attributs
    ---------
        blue(ndarray) : contient les valeurs de la bande bleue de l'image
        red(ndarray) : contient les valeurs de la bande rouge de l'image
        ...

    """

    def __init__(self, imgPath, zone, sensor = "LS8"):
        """
        description
        -----------
            Méthode appelée automatiquement lors de l'instanciation d'un objet Image.
            Elle permet d'initialiser tous les attributs d'instance.
        
        parameters
        ----------
            self(Image): désigne l'instance de la classe Image sur laquelle cette fonction va agir. (N'a pas à être listé les paramètres lors de l'appel)

            imgPath(str): chemin vers un dossier contenant toutes les bandes d'une image LS8
            
            zone(list): Contient les coordonnées du point haut-gauche et bas-droite d'une zone d'intérêt à charger.

            sensor(str): Permettrait de modifier le comportement de cette fonction pour adapter
            cette classe à d'autres types de produits, issus de capteurs différents.
        
        returns
        -------
            Cette fonction ne renvoie rien mais permet de charger des données dans les attributs
            d'instance de type Image.

        """

        # Création d'un dictionnaire dans lequel on va stocker les différentes bandes
        bands = {}

        # Unpack de la liste zone transmise en paramètres
        xMin, xMax, yMin, yMax = zone

        # Vérification de la validité logique des coordonnées
        if xMin >= xMax or yMin >= yMax :
            raise ValueError("Coordonnées de la zone d'intérêt invalides")

        # Information de l'utilisateur du chargement
        print("Recherche et chargement des bandes spectrales")

        # Parcours de tous les fichiers dans le dossier de l'image
        for fichier in os.listdir(imgPath):
            
            # Si c'est une image .tif
            if fichier.upper().endswith(".TIF"):

                try:
                    # Elaboration d'un motif de chaîne par expression régulière

                    motif = re.compile("B[0-9]+.TIF$") # [0-9]+ = un ou plusieurs chiffres
                                                       # $ = fin de la chaîne

                    # Recherche de la position de début de ce motif dans le nom de l'image
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

                # AttributeError peut être soulevée lors la transformation du ds en geotransform
                # dans le cas où le nom de l'image ne comporte pas de chaîne correspondant au motif recherché (Ca veut dire que c'est pas une bande)
                
                except AttributeError:
                    # On informe que le fichier n'a pas été chargé
                    print(">>> Fichier {} ignoré".format(fichier))

        # Transfère les éléments du dictionnaire bands dans des attributs d'instance
        if sensor == "LS8":
            self.blue = bands["B2"]
            self.green = bands["B3"]
            self.red = bands["B4"]
            self.nir = bands["B5"]
            self.swir1 = bands["B6"]
            self.swir2 = bands["B7"]

        return None

    def compute_ci1(self):
        """
        Calcule l'indice nuageux ci1 à partir des bandes de l'image
        """
        ci1 = (self.blue + self.green + self.red)/(self.nir + self.swir1*2)
        return ci1

    def compute_ci2(self):
        """
        Calcule l'indice nuageux ci2 à partir des bandes de l'image
        """
        ci2 = (self.blue + self.green + self.red + self.nir + self.swir1 + self.swir2) / 6
        return ci2

    def classifIndice(self, matriceIndice, mode, T1 = 0.1, t2 = 0.1):
        """
        description
        -----------
            Produit une classification binaire à partir d'une matrice et d'un seuil

        parameters
        ----------
            matriceIndice(ndarray) : matrice à classifier
            mode(int) : Modifie le comportement de la fonction selon la classif à réaliser
                        Modalité : mode = 1 =====> Classification CI1
                                   mode = 2 =====> Classification CI2
        
        returns
        -------
            classif(ndarray) : matrice binaire ayant des valeurs 0 et 255.
            Les valeurs 0 correspondent aux pixels qui ne respectent pas la condition.
            Les valeurs 255 à ceux qui la respectent.
            0 et 255 permet d'avoir des images en noir et blanc avec pillow.show()
        """
        if mode == 1:
            classif = np.where(matriceIndice < T1, 255, 0)

        elif mode == 2:

            # Définition du seuil T2 en fonction du paramètre t2
            meanci2 = np.mean(matriceIndice[~np.isnan(matriceIndice)])
            maxci2 = np.max(matriceIndice[~np.isnan(matriceIndice)])
            T2 = meanci2 + (t2 * (maxci2 - meanci2))

            # Classification
            classif = np.where(matriceIndice > T2, 255, 0)
        else:
            raise ValueError("Mode inconnu")

        return classif

    def fusionClassifs(self, matrice1, matrice2):
        """
        Additionne deux matrices.
        """
        return matrice1 + matrice2

    def surfaceNonZero(self, matrice):
        """
        description
        -----------
            Calcule en pourcentage
            Comptabilise les pixels ayant non nulle au sein de la matrice, diviser par la surface totale de la matrice.
        
        parameters
        ----------
            matrice (ndarray) : Matrice 
        
        """
        return (np.count_nonzero(matrice)/(np.shape(matrice)[0]*np.shape(matrice)[1]))*100 # np.count_nonzero() nous retourne un int indiquant 
                                                                                            # le nombre de valeur différentde zéro
                                                                                            # l'expression: np.shape(matrice)[0]*np.shape(matrice)[1] in dique la superficie de la matrice
                                                                                            # np.shape() renvoie un liste ayant comme contenue les dimension (X,Y) de notre matrice
                                                    

    def matrice2png(self, matrice, mode = 0, path = ""):
        """
        description
        -----------
            Convertit une matrice ndarray en image
        
        parameters
        ----------
            matrice(ndarray) : matrice à convertir
            mode(int) : permet d'adapter le comportement pour enregistrer une image ou simplement l'afficher
            path(str) : nom de fichier lors de la sauvegarde de l'image
        """
        if mode == 0:
            pillow_img.fromarray(matrice).convert("L").show()
        
        elif mode == 1:
            print("Ecriture disque de la matrice de fusion")
            
            try:
                pillow_img.fromarray(matrice).convert("L").save(path)

            except ValueError:
                print("Chemin d'enregistrement de l'image inconnu")
        
        else:
            print("Mode inconnu")
        
        return None

