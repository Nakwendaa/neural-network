# coding=utf-8
# Paul Chaffanet et Émile Labbé

import gzip
import pickle
import time
import os
import numpy as np
import matplotlib.pyplot as plt
from Mlp import Mlp as Mlp, MlpMat as MlpMat


def ratio(mat1, mat2):
    """
    Effectue le calcul du ratio entre deux ndarray de même dimension

    :param mat1: ndarray
    :param mat2: ndarray est de même dimension que mat1

    :return: ndarray de même dimension que mat1 et mat 2 contenant les
    ratios
    """
    ratios = []
    for elem1, elem2 in zip(np.nditer(mat1), np.nditer(mat2)):
        ratios.append(1.0 if elem1 == elem2 else elem1 / elem2)
    return np.array(ratios).reshape(mat1.shape)


# Principale fonction des exercices 1, 2, 3 et 4.
def check_grad(data, k):
    """
    Cette fonction permet d'effectuer la vérification de notre implémentation du
    gradient sur Mlp:
        1) On calcule tout d'aborder le gradient des paramètres actuels du
           réseau par rétropropagation sur le batch.
        2) On calcule ensuite le gradient par différences finies avec les mêmes
           paramètres actuels
        3) On effectue le ratio entre le gradient calculé à l'étape (1) et ce-
           lui de l'étape (2) afin de pouvoir les comparés.

    Si les ratios calculés sont entre 0.99 et 1.01, cela signifie que le calcul
    de notre gradient par rétropagation est correct.

    :param data: ndarray de dimension n * (d + 1) avec la dernière colonne
    contenant les labels. Un batch aléatoire de taille K sera pioché dans data.
    :param k: la taille du batch pour lequel on souhaite effectuer une vérifica-
    tion du gradient

    """

    # Instantiation d'un Mlp avec d neurones (x étant de taille d dans data)
    # et m neurones (nombre de classes uniques dans data, donc si deux classes
    # alors m = 2)
    mlp = Mlp(d=data.shape[1] - 1, dh=2, m=np.unique(data[:, -1]).size,
              wd=np.zeros((4, 1)), k=10)
    # Mélange data
    np.random.shuffle(data)
    # Sélectionne les K premières lignes afin de constituer un batch de taille K
    batch = data[0:k, :]
    # Gradient calculé par rétropropagation sur le batch avec les paramètres
    # du réseau
    grad_w1, grad_b1, grad_w2, grad_b2 = mlp.compute_grad(batch)
    # Gradient calculé par différence finie sur le batch avec les paramètres
    # du réseau
    f_grad_w1, f_grad_b1, f_grad_w2, f_grad_b2 = mlp.compute_finite_difference(
                                                                          batch)

    # Calcul du ratio pour chaque paramètre
    ratio_w1 = ratio(grad_w1, f_grad_w1)
    ratio_w2 = ratio(grad_w2, f_grad_w2)
    ratio_b1 = ratio(grad_b1, f_grad_b1)
    ratio_b2 = ratio(grad_b2, f_grad_b2)

    # Affichage des ratios pour chaque paramètres
    line = ""
    line += "\tRatios W1\n---------------\n"
    line += np.array_str(ratio_w1) + "\n\n\n"
    line += "\tRatios b1\n---------------\n"
    line += np.array_str(ratio_b1) + "\n\n\n"
    line += "\tRatios W2\n---------------\n"
    line += np.array_str(ratio_w2) + "\n\n\n"
    line += "\tRatios b2\n---------------\n"
    line += np.array_str(ratio_b2) + "\n\n\n"

    # Vérification de la conformité des ratios
    if (ratio_w1 >= 0.99).all() and (ratio_w1 <= 1.01).all() \
            and (ratio_b1 >= 0.99).all() and (ratio_b1 <= 1.01).all() \
            and (ratio_w2 >= 0.99).all() and (ratio_w2 <= 1.01).all() \
            and (ratio_b2 >= 0.99).all() and (ratio_b2 <= 1.01).all():
        line += "Vérification du gradient réussie pour K = "+str(k) + "\n\n"
    else:
        line += "Vérification du gradient échouée pour K = "+str(k) + "\n\n"
    print (line)


def plot_area(net, batch, title, file_name, directory = None):

    """

    :param net: une référence au réseau MLP afin de pouvoir calculer des prédic-
                tions à partir d'un batch
    :param batch: un batch de dimension n * 3 contenant le jeu de données
                  que l'on souhaite afficher dans le graphique
    :param title: le titre du graphique
    :param file_name: le nom du fichier
    :param directory: le nom du répertoire où l'on souhaite écrire notre fichier
    """
    batch_x = np.array(batch[:, :net.d])
    batch_y = np.array(batch[:, -1])

    x1_lin = np.linspace(min(batch_x[:, 0]) - 0.5, max(batch_x[:, 0]) + 0.5,
                         num=500)
    x2_lin = np.linspace(min(batch_x[:, 1]) - 0.5, max(batch_x[:, 1]) + 0.5,
                         num=500)
    # Créer une grille de points
    grid = np.array([[x1, x2] for x2 in x2_lin for x1 in x1_lin])
    # Calculer les prédictions pour ceux-ci afin de dessiner les régions
    # de décisions
    predictions = net.compute_predictions(grid)
    plt.scatter(grid[:, 0], grid[:, 1], s=25, c=predictions, marker=".",
                alpha=0.01)
    plt.scatter(batch_x[:, 0], batch_x[:, 1], c=batch_y, marker='v', s=25,
                edgecolors="black")

    plt.title(title)
    if directory is not None:
        plt.savefig(directory + "/" + file_name + ".png")
    else:
        plt.savefig(file_name + ".png")
    print("Nouveau fichier : " + file_name + ".png")
    plt.close()


def plot_curve(train, valid, test, x, title, file_name, directory=None):
    """

    :param train: une liste d'erreurs/coûts d'entraînement
    :param valid: une liste d'erreurs/coûts de validation
    :param test: une liste d'erreurs/coûts d'entraînement
    :param x: une liste d'époques de 1 à n (axe des x)
    :param title: le titre du graphique
    :param file_name: le nom du fichier à créer
    :param directory: le nom du répertoire où l'on souhaite écrire notre fichier
    """
    plt.plot(x, train, label="Train")
    plt.plot(x, valid, label="Validation")
    plt.plot(x, test, label="Test")
    plt.legend()
    plt.title(title)
    if directory is not None:
        plt.savefig(directory + "/" + file_name + ".png")
    else:
        plt.savefig(file_name + ".png")
    print("Nouveau fichier : " + file_name + ".png")
    plt.close()


def exercice5(data_moon):
    # Création des dossiers pour recesoir les nombreux graphiques
    current_directory = os.getcwd()
    final_directory = os.path.join(current_directory, r'exo5_part1')
    if not os.path.exists(final_directory):
        os.makedirs(final_directory)
    final_directory = os.path.join(current_directory, r'exo5_part2')
    if not os.path.exists(final_directory):
        os.makedirs(final_directory)
    final_directory = os.path.join(current_directory, r'exo5_part3')
    if not os.path.exists(final_directory):
        os.makedirs(final_directory)
    final_directory = os.path.join(current_directory, r'exo5_part4')
    if not os.path.exists(final_directory):
        os.makedirs(final_directory)

    # Première série de tests d'hyper-paramètres
    np.random.shuffle(data_moon)
    d = data_moon.shape[1] - 1
    m = np.unique(data_moon[:, -1]).size
    # Différentes configurations d'hyper-paramètres
    dh = 20
    wd_a = [[0, 0, 0, 0], [0.1, 0, 0, 0], [0, 0.1, 0, 0], [0.1, 0.1, 0, 0],
            [0.5, 0, 0, 0], [0, 0.5, 0, 0], [0.5, 0.5, 0, 0], [0, 0, 0.1, 0],
            [0, 0, 0, 0.1], [0, 0, 0.1, 0.1], [0, 0, 0.5, 0], [0, 0, 0, 0.5],
            [0, 0, 0.5, 0.5], [0.1, 0.1, 0.1, 0.1], [0.5, 0.5, 0.5, 0.5]]
    k = 25
    epoch = 25
    learning_rate = 0.1

    for wd in wd_a:
        mlp = Mlp(d, dh, m, wd, k)
        error = mlp.train(training_data=data_moon, epochs=epoch,
                          learning_rate=learning_rate)
        title = "Error = " + str(error) + "%, d = " + str(d)
        title += ", dh = " + str(dh) + ", m = " + str(m) + "\nepochs = "
        title += str(epoch) + ", wd = " + str(wd) + ", K = " + str(k)
        title += ", l_rate = " + str(learning_rate)
        file_name = "dh_ " + str(dh) + "_m_" + str(m) + "_epochs_"
        file_name += str(epoch) + "_wd_" + str(wd) + "_K_" + str(k)
        file_name += "_l_rate_" + str(learning_rate)
        plot_area(mlp, data_moon, title, file_name, directory="exo5_part1")

    # Deuxième série de tests d'hyper-paramètres
    np.random.shuffle(data_moon)
    d = data_moon.shape[1] - 1
    m = np.unique(data_moon[:, -1]).size
    dh_a = [2, 20, 200]
    wd = [0.05, 0.05, 0, 0]
    k = 25
    epoch_a = [10, 25, 50, 100]
    learning_rate = 0.1
    for dh in dh_a:
        for epoch in epoch_a:
            mlp = Mlp(d, dh, m, wd, k)
            error = mlp.train(training_data=data_moon, epochs=epoch,
                              learning_rate=learning_rate)
            title = "Error = " + str(error) + "%, d = " + str(d)
            title += ", dh = " + str(dh) + ", m = " + str(m) + "\nepochs = "
            title += str(epoch) + ", wd = " + str(wd) + ", K = " + str(k)
            title += ", l_rate = " + str(learning_rate)
            file_name = "dh_ " + str(dh) + "_m_" + str(m) + "_epochs_"
            file_name += str(epoch) + "_wd_" + str(wd) + "_K_" + str(k)
            file_name += "_l_rate_" + str(learning_rate)
            plot_area(mlp, data_moon, title, file_name, directory="exo5_part2")

    # Troisième série de tests d'hyper-paramètres
    np.random.shuffle(data_moon)
    d = data_moon.shape[1] - 1
    m = np.unique(data_moon[:, -1]).size
    dh = 20
    wd = [0.05, 0.05, 0, 0]
    k_list = [1, 5, 10, 25]
    epoch_a = [25, 50, 100]
    learning_rate_a = [0.01, 0.05, 0.1, 0.5]
    for epoch in epoch_a:
        for k in k_list:
            for learning_rate in learning_rate_a:
                mlp = Mlp(d, dh, m, wd, k)
                error = mlp.train(training_data=data_moon, epochs=epoch,
                                  learning_rate=learning_rate)
                title = "Error = " + str(error) + "%, d = " + str(d)
                title += ", dh = " + str(dh) + ", m = " + str(m) + "\nepochs = "
                title += str(epoch) + ", wd = " + str(wd) + ", K = " + str(k)
                title += ", l_rate = " + str(learning_rate)
                file_name = "dh_ " + str(dh) + "_m_" + str(m) + "_epochs_"
                file_name += str(epoch) + "_wd_" + str(wd) + "_K_" + str(k)
                file_name += "_l_rate_" + str(learning_rate)
                plot_area(
                    mlp, data_moon, title, file_name, directory="exo5_part3")

    # 4ème série de tests d'hyper-paramètres
    np.random.shuffle(data_moon)
    d = data_moon.shape[1] - 1
    m = np.unique(data_moon[:, -1]).size
    dh = 20
    wd = [0, 0, 0, 0]
    k_list = [1, 5, 10, 25]
    epoch_a = [25, 50, 100]
    learning_rate_a = [0.01, 0.05, 0.1, 0.5]
    for epoch in epoch_a:
        for k in k_list:
            for learning_rate in learning_rate_a:
                mlp = Mlp(d, dh, m, wd, k)
                error = mlp.train(training_data=data_moon, epochs=epoch,
                                  learning_rate=learning_rate)
                title = "Error = " + str(error) + "%, d = " + str(d)
                title += ", dh = " + str(dh) + ", m = " + str(m) + "\nepochs = "
                title += str(epoch) + ", wd = " + str(wd) + ", K = " + str(k)
                title += ", l_rate = " + str(learning_rate)
                file_name = "dh_ " + str(dh) + "_m_" + str(m) + "_epochs_"
                file_name += str(epoch) + "_wd_" + str(wd) + "_K_" + str(k)
                file_name += "_l_rate_" + str(learning_rate)
                plot_area(
                    mlp, data_moon, title, file_name, directory="exo5_part4")


def main():

    np.random.seed(333)
    # Chargement des données des deux lunes
    data_moon = np.loadtxt(open('2moons.txt', 'r'))
    # Chargement des données MNIST
    data_mnist = pickle.load(gzip.open('mnist.pkl.gz'))

    """
    
    Exercice 1 : Calculer le gradient pour un exemple et v́erifiez que le calcul
    est correct avec la technique de v́erification du gradient par diff́erence
    finie.

    Exercice 2: Produire un affichage de verification du gradient par difference 
    finie pour votre ŕeseau (pour un petit ŕeseau, par ex. d = 2 et dh = 2 
    initialisé aĺeatoirement) sur 1 exemple.
    
    """

    print ("***************************************************************"
           "*****************************************\n")
    print ("EXERCICE 1 ET 2\n\n")
    check_grad(data=data_moon, k=1)

    """
    Exercice 3: Ajoutez à cette version un hyperparamètre de taille de lot K, 
    pour permettre le calcul du gradient par mini-lot de K exemples (présentés 
    sous forme de matrices), en faisant une boucle sur les K exemples (c’est 
    un petit ajout à votre code précédent).

    Exercice 4: Produire un affichage de verification du gradient sur les 
    paramètres, par différence finie pour votre réseau (pour un petit réseau, 
    par ex. d = 2 et dh = 2 initialisé aléatoirement) pour un lot de 10 exemples 
    (vous pouvez prendre des exemples des deux classes du jeux de donnée des 2 
    lunes).
    """

    print ("***************************************************************"
           "*****************************************\n")
    print ("EXERCICE 3 ET 4\n\n")
    check_grad(data=data_moon, k=10)

    """
    Exercice 5: Entrainer votre ŕeseau de neurones par descente de gradient 
    sur les donnnées du problème des deux-lunes. Afficher les régions de 
    décision pour différentes valeurs d’hyper-paramètres (weight decay, nombre 
    d’unites cach́ees, arrêt prématuré) de façon à illustrer leur effet sur le 
    contrôle de capacit́e.
    """

    # L'exercice 5 est mis en commentaire car il prend un certain à produire
    # tous les graphiques. Enlever # pour activer la fonction
    # exercice5(data_moon) afin de pouvoir tester la fonction

    print ("***************************************************************"
           "*****************************************\n")
    print ("EXERCICE 5\n\n")
    # En commentaire car prend beaucoup de temps à produire tous les graphes
    # exercice5(data_moon)

    """
    Exercice 7: Comparez vos deux implémentations (avec et sans boucle sur les 
    exemples du lot)  pour v́erifier qu’elles donnent le même gradient total sur 
    les paramètres, d’abord avec K = 1. Puis comparez-les avec K = 10.
    Joignez à votre rapport les affichages numériques effectués pour cette 
    comparaison.
    """

    print ("***************************************************************"
           "*****************************************\n")
    print ("EXERCICE 7\n\n")

    d = 2
    dh = 2
    m = 2
    wd = [0, 0.2, 0, 0.1]

    # Gradient sur paramètres avec K = 1
    k = 1
    # Mélange data (in-place)
    np.random.shuffle(data_moon)
    # Sélectionne les K premières lignes afin de constituer un batch de taille K
    batch = data_moon[0:k, :]

    mlp = Mlp(d, dh, m, wd, k=50)
    mlp_mat = MlpMat(d, dh, m, wd, k=50, mlp=mlp)
    mlp_grad_w1, mlp_grad_b1, mlp_grad_w2, mlp_grad_b2 = mlp.compute_grad(batch)
    mlp_mat_grad_w1, mlp_mat_grad_b1, mlp_mat_grad_w2, mlp_mat_grad_b2 = mlp_mat.compute_grad(batch)

    ratio_w1 = 1 - abs(ratio(mlp_mat_grad_w1, mlp_grad_w1))
    ratio_w2 = 1 - abs(ratio(mlp_mat_grad_w2, mlp_grad_w2))
    ratio_b1 = 1 - abs(ratio(mlp_mat_grad_b1, mlp_grad_b1))
    ratio_b2 = 1 - abs(ratio(mlp_mat_grad_b2, mlp_grad_b2))
    if len(ratio_w1[ratio_w1 > 1e-10]) == 0 and len(
           ratio_w2[ratio_w2 > 1e-10]) == 0 and len(
           ratio_b1[ratio_b1 > 1e-10]) == 0 and len(
           ratio_b2[ratio_b2 > 1e-10]) == 0:
        print ("K = 1: Gradients identiques")
    else:
        print ("K = 1: Gradient pas identiques")

    # Gradient sur paramètres avec K = 10
    k = 10
    # Mélange data (in-place)
    np.random.shuffle(data_moon)
    # Sélectionne les K premières lignes afin de constituer un batch de taille K
    batch = data_moon[0:k, :]
    mlp = Mlp(d, dh, m, wd, k=50)
    mlp_mat = MlpMat(d, dh, m, wd, k=50, mlp=mlp)
    mlp_grad_w1, mlp_grad_b1, mlp_grad_w2, mlp_grad_b2 = mlp.compute_grad(batch)
    mlp_mat_grad_w1, mlp_mat_grad_b1, mlp_mat_grad_w2, mlp_mat_grad_b2 = mlp_mat.compute_grad(batch)
    ratio_w1 = 1 - abs(ratio(mlp_mat_grad_w1, mlp_grad_w1))
    ratio_w2 = 1 - abs(ratio(mlp_mat_grad_w2, mlp_grad_w2))
    ratio_b1 = 1 - abs(ratio(mlp_mat_grad_b1, mlp_grad_b1))
    ratio_b2 = 1 - abs(ratio(mlp_mat_grad_b2, mlp_grad_b2))
    if len(ratio_w1[ratio_w1 > 1e-10]) == 0 and len(
           ratio_w2[ratio_w2 > 1e-10]) == 0 and len(
           ratio_b1[ratio_b1 > 1e-10]) == 0 and len(
           ratio_b2[ratio_b2 > 1e-10]) == 0:
        print ("K = 10: Gradients identiques")
    else:
        print ("K = 10: Gradient pas identiques")

    """
    Exercice 8: Mesurez le temps que prend une ́epoque sur MNIST 
    (1  ́epoque = 1 passage complet à travers l’ensemble d’entraînement) pour 
    K = 100 avec chacune des deux implémentations (mini-lot par boucle, et 
    mini-lot avec calcul matriciel).
    """

    print ("***************************************************************"
           "*****************************************\n")
    print ("EXERCICE 8\n\n")

    # Sélection de l'ensemble d'entraînement de MNIST
    train_x = data_mnist[0][0]
    train_y = data_mnist[0][1]
    train_y.shape = (1, train_y.shape[0])
    train = np.concatenate((train_x, np.transpose(train_y)), axis=1)
    np.random.shuffle(train)

    # Hyper-paramètres
    d = train_x.shape[1]
    m = np.unique(train_y).size
    dh = 10
    wd = [0, 0, 0, 0]
    k = 50
    epoch = 1
    learning_rate = 0.1

    # MLP par boucle
    mlp = Mlp(d, dh, m, wd, k)
    start_time = time.time()
    mlp.train(train, epoch, learning_rate)
    print ("--- %s secondes --- pour 1 époque de MLP avec boucle."
           % (time.time() - start_time))

    # MLP par calcul matriciel
    mlp_mat = MlpMat(d, dh, m, wd, k, mlp)
    start_time = time.time()
    mlp_mat.train(train, epoch, learning_rate)
    print ("--- %s secondes --- pour 1 époque de  MLP avec calcul matriciel."
           % (time.time() - start_time))

    """
    Exercice 9: Adaptez votre code pour qu’il calcule au vol, pendant 
    l’entraînement, l’erreur de classification totale sur l’ensemble 
    d’entraînement, en plus du coût optimisé total (somme des L encourus), ceci 
    pour chaque  ́epoque d’entraînement, et qu’après chaque  ́epoque 
    d’entraînement, il calcule aussi erreur et coût moyen sur l’ensemble de 
    validation et de test. Faites en sorte qu’il les affiche après chaque  
    ́epoque les 6 nombres correspondants (erreur et coût moyen sur train, 
    valid, test) et les ́ecrive dans un fichier.
    """
    """
    Exercice 10: Entrainer votre réseau sur les données de MNIST. Produisez 
    les courbes d’entraînement, de validation et de test (courbes de l’erreur de 
    classification et du coût en fonction du nombre d’́epoques d’entraînement, 
    qui correspondent à ce que vous avez enregistŕe dans un fichier à la 
    question précédente). 
    Joignez à votre rapport les courbes obtenues avec votre meilleure valeur 
    d’hyper-paramètres, c.à.d. pour lesquels vous avez atteint la plus basse 
    erreur de classification sur l’ensemble de validation. 
    On suggère deux graphiques : un pour les courbes de taux d’erreurs de 
    classification (train, valid, test avec des couleurs diff́erentes, bien 
    pŕeciśees dans la ĺegende) et l’autre pour la perte moyenne (le L moyen 
    sur train, valid , test). 
    Normalement vous devriez pouvoir atteindre moins de 5% d’erreur en test. 
    Indiquez dans votre rapport la valeur des hyper-paramètres retenue et 
    correspondant aux courbes que vous joignez.
    """
    print ("***************************************************************"
           "*****************************************\n")
    print ("EXERCICE 9 ET 10\n\n")

    # Cette partie prend énormément de temps à s'exécuter (donc nous l'avons mis
    # en commentaire). Nous avons testé des configurations différentes
    # d'hyperparamètres de manière intuitive par rapport à ce que nous avons
    # constaté de l'exercice 5 en ce qui concerne leur comportement.
    # On a alors gardé la valeur des hyper-paramètres pour lequel nous avons
    # eu la plus basse erreur de validation et nous avons produit les courbes
    # d'erreur pour cette configuration d'hyper-paramètres

    """
    d = data_mnist[0][0].shape[1]
    m = np.unique(data_mnist[0][1]).size
    dh = d
    wd_a = [[0, 0, 0, 0], [0, 0.05, 0, 0.05], [0.05, 0.05, 0, 0],
            [0, 0, 0.05, 0.05]]
    k_a = [32, 64, 128, 256]
    epochs = [50, 75, 100]
    learning_rate_a = [0.01, 0.05, 0.1, 0.25]
    res = []
    for learning_rate in learning_rate_a:
        for k in k_a:
            for wd in wd_a:
                for epoch in epochs:

                    train_x = np.array(data_mnist[0][0])
                    train_y = np.array(data_mnist[0][1])
                    train_y.shape = (1, train_y.shape[0])
                    valid_x = np.array(data_mnist[1][0])
                    valid_y = np.array(data_mnist[1][1])
                    valid_y.shape = (1, valid_y.shape[0])
                    training_data = np.concatenate((train_x,
                                                   np.transpose(train_y)),
                                                   axis=1)

                    mlp_mat = MlpMat(d, dh, m, wd, k)
                    mlp_mat.train(training_data, epoch, learning_rate)

                    predictions = mlp_mat.compute_predictions(valid_x)
                    diff = predictions - valid_y
                    valid_err = len(diff[diff != 0]) * 100. / valid_x.shape[0]

                    res.append((valid_err,
                                [d, dh, m, wd, k, epoch, learning_rate]))
                    print (str(res[len(res) - 1])) + " testé"

    # Trouver la meilleure configuration d'hyper-paramètres
    res = min(res)
    print "\nMeilleure configuration d'hyperparamètres:"
    print ("d = " + str(res[1][0]) + ", dh = " + str(
        res[1][1]) + ", m = " +
           str(res[1][2]) + ", wd = " + str(
        res[1][3]) + ", K = " +
           str(res[1][4]) + ", epoch = " + str(
        res[1][5]) + ", learning_rate = "
           + str(res[1][6]),
           "valid_error = " + str(res[0]) + "%")
    """

    d = data_mnist[0][0].shape[1]
    m = np.unique(data_mnist[0][1]).size
    dh = data_mnist[0][0].shape[1]
    # On régularise que très légèrement (on a besoin d'une bonne capacité
    # pour réduire l'erreur)
    wd = [0, 0.005, 0, 0.005]
    # Taille de batch
    k = 32
    # Convergence assez forte à partir de l'époque 25
    epoch = 50
    learning_rate = 0.05

    mlp_mat = MlpMat(d, dh, m, wd, k)
    err_loss = mlp_mat.train(data_mnist, epoch, learning_rate,
                             is_mnist=True)
    train_err = [line[0] for line in err_loss]
    valid_err = [line[1] for line in err_loss]
    test_err = [line[2] for line in err_loss]
    train_avg_loss = [line[3] for line in err_loss]
    valid_avg_loss = [line[4] for line in err_loss]
    test_avg_loss = [line[5] for line in err_loss]

    # Écriture dans un fichier des erreurs et coûts par époque
    file_csv = open("error_and_cost_mlp_mat.csv", 'w')
    file_csv.write("NoEpoch, TrainErr, ValidErr, TestErr, TrainAvgLoss, "
                   "ValidAvgLoss, TestAvgLoss\n")
    for i in range(len(err_loss)):
        file_csv.write(str(i + 1) + ", " + str(err_loss[i][0]) + "%, " +
                       str(err_loss[i][1]) + "%, " + str(err_loss[i][2]) + "%, "
                       + str(err_loss[i][3]) + ", " + str(err_loss[i][4]) + ", "
                       + str(err_loss[i][5]) + "\n")
    file_csv.close()
    print("Nouveau fichier : error_and_cost_mlp_mat.csv")

    # Plot des courbes d'erreur, de test et de validation en fonction de l'épo-
    # que
    title = "Erreur en % - " + str(epoch) + " epoques"
    name = str(dh) + "_" + str(wd) + "_" + str(dh) + "_" + "erreur"
    plot_curve(train_err, valid_err, test_err, range(1, epoch + 1), title, name)

    title = "Cout moyen - " + str(epoch) + " epoques"
    name = str(dh) + "_" + str(wd) + "_" + str(dh) + "_" + "cout_moyen"
    plot_curve(train_avg_loss, valid_avg_loss, test_avg_loss,
               range(1, epoch + 1), title, name)


main()
