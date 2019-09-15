# coding=utf-8
# Paul Chaffanet et Émile Labbé

import numpy as np

np.random.seed(647)


class Mlp:

    def __init__(self, d, dh, m, wd, k):
        """

        :param d: Nombre de neurones de la couche d'entrée (int).
        :param dh: Nombre de neurones de la couche cachée (int).
        :param m: Nombre de neurones de la couche de sortie (int).
        :param wd: une liste de 4 float représentant les weight-decays
                   [λ_11, λ_12, λ_21, λ_22].
        :param k: un entier représentant la taille du mini-batch (int).
        """
        self.d = d
        self.dh = dh
        self.m = m
        self.wd = wd
        self.K = k

        # Initialiser de manière aléatoire les poids selon une loi uniforme
        # bornée par -1.0/sqrt(nc) et 1.0/sqrt(nc) où nc est le nombre d'entrées
        # de la couche d'entrée du poids
        # Ici, nc = d
        self.W1 = np.random.uniform(low=-1.0/np.sqrt(d), high=1.0/np.sqrt(d),
                                    size=(dh, d))
        # Pour W2, nc = dh neurones dans la couche d'entrée
        self.W2 = np.random.uniform(low=-1.0/np.sqrt(dh), high=1.0/np.sqrt(dh),
                                    size=(m, dh))

        # Initialiser les biais à 0
        self.b1 = np.zeros((dh, 1))
        self.b2 = np.zeros((m, 1))

    def fprop(self, x):
        """
        Propagation avant d'un exemple x.

        :param x: ndarray de dimension 1 * d.

        :return: un tuple (ha,hs,oa,os) de ndarray de dimensions respectives
                (dh * 1, dh * 1, m * 1, m * 1).
        """
        # Transposer x pour obtenir un ndarray de dimension d * 1
        x = np.transpose(x)
        # W1 est de dimension dh * d, x est de dimension  d * 1, b1 est de
        # dimension dh * 1 donc ha est de dimension dh * 1
        ha = np.dot(self.W1, x) + self.b1
        # hs est de dimension dh * 1 après passage de ha dans la fonction
        # rect
        hs = rect(ha)
        # W2 est de dimension m * dh, hs est de dimension  dh * 1, b2 est de
        # dimension m * 1 donc oa est de dimension m * 1
        oa = np.dot(self.W2, hs) + self.b2
        # os est de dimension m * 1 après passage de oa dans la fonction
        # softmax
        os = softmax(oa)
        return ha, hs, oa, os

    def bprop(self, x, y):
        """
        Propagation arrière d'un exemple (x,y).

        :param x: un ndarray de dimension 1 * d.
        :param y: un int représentant la classe de x.

        :return: un tuple (grad_w1, grad_b1, grad_w2, grad_b2) de ndarray de
                 dimensions respectives (dh * d, dh * 1, m * dh, m * 1)
                 représentant le gradient de la fonction de perte de l'exemple
                 (x,y) avec les paramètres actuels.
        """
        # un tuple (ha,hs,oa,os) de ndarray de dimensions respectives
        # (dh * 1, dh * 1, m * 1, m * 1) après propagation avant (afin de
        # récupérer les valeurs de (ha, hs, oa, os) pour cet exemple x)
        ha, hs, oa, os = self.fprop(x)
        # Transposer x afin d'obtenir un vecteur colonne d * 1
        x = np.transpose(x)
        # grad_b2 et grad_oa sont de dimension m * 1
        grad_b2 = grad_oa = os - np.transpose(onehot(np.array([[y]]), self.m))
        # grad_W2: avec grad_oa de dimension m * 1 et hs de dimension dh * 1
        # donc grad_W2 est de dimension m * dh
        grad_w2 = np.dot(grad_oa, np.transpose(hs))
        # Régularisation en ajoutant les weight-decays pour le gradient de W2
        grad_w2 += self.wd[2] + 2 * self.wd[3] * self.W2
        # grad_hs est de dimension dh * 1 car la transposée de W2 est de
        # dimension dh * m et grad_oa de dimension m * 1
        grad_hs = np.dot(np.transpose(self.W2), grad_oa)
        # grad_b1 et grad_ha sont de dimension dh * 1
        grad_b1 = grad_ha = grad_hs * np.where(ha > 0, [1], [0])
        # grad_ha est de dimension dh * 1 et x_t de dimension  1 * d
        # alors grad_w1 et dh * d
        grad_w1 = np.dot(grad_ha, np.transpose(x))
        # Régularisation en ajoutant les weight-decays pour le gradient de W1
        grad_w1 += self.wd[0] + 2 * self.wd[1] * self.W1
        return grad_w1, grad_b1, grad_w2, grad_b2

    def compute_loss(self, x, t):
        """
        Calcule la perte pour chaque exemple d'entraînement x.

        :param x: ndarray de dimension 1 * d qui correspond aux entrées.
        :param t: entier qui correspond à la cible (classe).

        :return: float correspondant au coût calculé pour le i-ème exemple.
        """

        # Propagation avant de x avec les paramètres actuels
        os = self.fprop(x)[3]
        return -np.log(os[t][0])

    def compute_grad(self, batch):
        """
        Calcule le gradient avec les paramètres actuels pour ce batch.

        :param batch: un ndarray de dimension n * d + 1 avec les labels contenus
                      dans la dernière colonne.
        :return: Un tuple (grad_w1, grad_b1, grad_w2, grad_b2) représentant
                 le gradient de la fonction de perte pour ce batch.
        """
        # batch est une matrice n * d + 1
        grad_w1 = np.zeros(self.W1.shape)
        grad_w2 = np.zeros(self.W2.shape)
        grad_b1 = np.zeros(self.b1.shape)
        grad_b2 = np.zeros(self.b2.shape)

        # Obtenir une matrice n * d
        batch_x = np.array(batch[:, :self.d])
        # Obtenir un array 1-D de labels
        batch_y = np.array(batch[:, -1])

        for x, y in zip(batch_x, batch_y):
            x = np.array(x).reshape(1, self.d)
            y = int(y)
            d_grad_w1, d_grad_b1, d_grad_w2, d_grad_b2 = self.bprop(x, y)
            # Comme l'on ajoute la régularisation à chaque appel à bprop,
            # on retranche cette valeur dans la boucle pour ne pas ajouter
            # la régularisation n fois (n étant le nombre d'exemples dans le
            # batch)
            grad_w1 += d_grad_w1 - (self.wd[0] + 2 * self.wd[1] * self.W1)
            grad_b1 += d_grad_b1
            grad_w2 += d_grad_w2 - (self.wd[2] + 2 * self.wd[3] * self.W2)
            grad_b2 += d_grad_b2

        # On ajoute la régularisation seulement à la fin avant de retourner le
        # résultat
        grad_w1 += self.wd[0] + 2 * self.wd[1] * self.W1
        grad_w2 += self.wd[2] + 2 * self.wd[3] * self.W2
        return grad_w1, grad_b1, grad_w2, grad_b2

    def compute_finite_difference(self, batch):
        """
        Calcule une estimation du gradient pour ce batch avec les paramètres
        actuels par différences finies.

        :param batch: un ndarray de dimension n * d + 1 avec les labels contenus
                      dans la dernière colonne.
        :return: Un tuple (grad_w1, grad_b1, grad_w2, grad_b2) représentant
                  une estimation du gradient pour ce batch avec les paramètres
                  actuels par différences finies.
        """
        current_cost = 0

        # Initialisation à 0 des matrices de coûts des paramètres modifiés
        # par epsilon.
        cost_w1 = np.zeros(self.W1.shape)
        cost_b1 = np.zeros(self.b1.shape)
        cost_w2 = np.zeros(self.W2.shape)
        cost_b2 = np.zeros(self.b2.shape)

        # Choix d'une petite valeur espilon choisie aléatoirement.
        epsilon = 1. / np.random.uniform(np.power(10, 4), np.power(10, 6))

        # Obtenir une matrice n * d.
        batch_x = np.array(batch[:, :self.d])
        # Obtenir une matrice n * 1.
        batch_y = np.array(batch[:, -1])

        for x, y in zip(batch_x, batch_y):

            x = np.array(x).reshape(1, self.d)
            y = int(y)

            # Calculer le coût de l'exemple (x,y) et l'ajouter au coût total.
            current_cost += self.compute_loss(x, y)

            # Pour chaque paramètre scalaire W1:
            for w, cost in zip(np.nditer(self.W1, op_flags=['readwrite']),
                               np.nditer(cost_w1, op_flags=['readwrite'])):
                # Modifier le paramètre scalaire par une petite valeur de
                # espilon.
                w[...] += epsilon
                # Calculer la perte pour l'exemple (x,y), puis l'ajouter à la
                # matrice de coût cost_w1.
                cost[...] += self.compute_loss(x, y)
                # Retrancher espsilon.
                w[...] -= epsilon

            for b, cost in zip(np.nditer(self.b1, op_flags=['readwrite']),
                               np.nditer(cost_b1, op_flags=['readwrite'])):
                b[...] += epsilon
                cost[...] += self.compute_loss(x, y)
                b[...] -= epsilon

            for w, cost in zip(np.nditer(self.W2, op_flags=['readwrite']),
                               np.nditer(cost_w2, op_flags=['readwrite'])):
                w[...] += epsilon
                cost[...] += self.compute_loss(x, y)
                w[...] -= epsilon

            for b, cost in zip(np.nditer(self.b2, op_flags=['readwrite']),
                               np.nditer(cost_b2, op_flags=['readwrite'])):
                b[...] += epsilon
                cost[...] += self.compute_loss(x, y)
                b[...] -= epsilon

        # Calcul du gradient par différences finies entre coûts modifiés
        # par epsilon, et coûts non modifiés.
        grad_w1 = (cost_w1 - current_cost) / epsilon
        grad_b1 = (cost_b1 - current_cost) / epsilon
        grad_w2 = (cost_w2 - current_cost) / epsilon
        grad_b2 = (cost_b2 - current_cost) / epsilon

        # Retourner l'estimation du gradient pour les paramètres actuels.
        return grad_w1, grad_b1, grad_w2, grad_b2

    def train(self, training_data, epochs, learning_rate, is_mnist=False):
        """
        Méthode d'entraînement pour le MLP.

        :param training_data: un ndarray de dimension n * (d + 1) avec la
                              la dernière colonne sauf si is_mnist=True,
                              auquel cas, on reçoit la matrice MNIST contenant
                              les ensembles d'entraînement, de validation et de
                              test.
        :param epochs: un int représentant le nombre d'époques (arrêt prématuré)
        :param learning_rate: un float représentant la taux d'apprentissage.
        :param is_mnist: ce paramètre permet de savoir si ce sont des données
                         MNIST qui sont reçues ou des données d'entraînement
                         simples.

        :return: l'erreur d'entraînement si c'est un simple ensemble d'entraî-
                 nement, un tuple (erreur d'entraînement, erreur de validation,
                 erreur de test) si is_mnist = true
        """
        err_loss = []
        train_x = train_y = valid_x = valid_y = test_x = test_y = np.array([])
        i = 0

        if is_mnist:
            train_x = np.array(training_data[0][0])
            print(train_x.shape)
            train_y = np.array(training_data[0][1])
            #train_y.shape = (1, train_y.shape[0])
            valid_x = np.array(training_data[1][0])
            valid_y = np.array(training_data[1][1])
            print(valid_y.shape)
            valid_y.shape = (1, valid_y.shape[0])
            test_x = np.array(training_data[2][0])
            test_y = np.array(training_data[2][1])
            test_y.shape = (1, test_y.shape[0])
            training_data = np.concatenate((train_x, np.transpose(train_y)),
                                           axis=1)

        n = len(training_data)

        for j in xrange(epochs):

            np.random.shuffle(training_data)
            # mini_batch de dimension K * d
            mini_batches = [training_data[k:k + self.K] for k in xrange(0, n,
                                                                        self.K)]

            for mini_batch in mini_batches:
                # Descente de gradient
                grad_w1, grad_b1, grad_w2, grad_b2 = self.compute_grad(
                                                                    mini_batch)
                self.W1 -= (learning_rate * 1. / len(mini_batch)) * grad_w1
                self.W2 -= (learning_rate * 1. / len(mini_batch)) * grad_w2
                self.b1 -= (learning_rate * 1. / len(mini_batch)) * grad_b1
                self.b2 -= (learning_rate * 1. / len(mini_batch)) * grad_b2

            if is_mnist:

                # Calcul de l'erreur
                predictions = self.compute_predictions(train_x)
                diff = predictions - train_y
                train_err = len(diff[diff != 0]) * 100. / train_x.shape[0]
                predictions = self.compute_predictions(valid_x)
                diff = predictions - valid_y
                valid_err = len(diff[diff != 0]) * 100. / valid_x.shape[0]
                predictions = self.compute_predictions(test_x)
                diff = predictions - test_y
                test_err = len(diff[diff != 0]) * 100. / test_x.shape[0]

                # Calcul du coût optimisé total
                train_y_t = np.transpose(train_y)
                l_train = self.compute_loss(train_x, train_y_t).sum()
                l_train /= len(train_x)

                valid_y_t = np.transpose(valid_y)
                l_valid = self.compute_loss(valid_x, valid_y_t).sum()
                l_valid /= len(valid_x)

                test_y_t = np.transpose(test_y)
                l_test = self.compute_loss(test_x, test_y_t).sum()
                l_test /= len(test_x)

                err_loss.append(
                    (train_err, valid_err, test_err, l_train, l_valid, l_test))

                print ("Epoch (" + str(i) + "): " + str(train_err) + "%, " + str(
                    valid_err) + "%, " + str(test_err) + "%, " + str(
                    l_train) + ", " + str(l_valid) + ", " + str(l_test))
                i += 1

        if is_mnist:
            return err_loss
        else:
            training_data_x = training_data[:, :self.d]
            training_data_y = training_data[:, -1]
            predictions = np.array(self.compute_predictions(training_data_x))
            diff = predictions - training_data_y
            train_err = len(diff[diff != 0]) * 100. / n
            return train_err

    def compute_predictions(self, test_data):
        """
        Fonction de prédiction pour le MLP.

        :param test_data: un ndarray de dimension n * d représentant les données
                          de test pour lesquelles on souhaite obtenir une liste
                          de prédictions.

        :return: une liste de sorties (de classes associées aux exemples x).
        """
        sorties = []
        for x in test_data:
            x = np.array(x).reshape(1, self.d)
            ha, hs, oa, os = self.fprop(x)
            sorties.append(np.argmax(os))

        return sorties


class MlpMat(Mlp):

    def __init__(self, d, dh, m, wd, k, mlp=None):
        Mlp.__init__(self, d, dh, m, wd, k)

        # Afin de copier les paramètres d'un MLP
        if mlp is not None:
            self.W1 = mlp.W1
            self.W2 = mlp.W2
            self.b1 = mlp.b1
            self.b2 = mlp.b2

    def fprop(self, train_x):
        """
        Propagation avant d'un ensemble d'entraînement x.

        :param train_x: ndarray de dimension n * d.

        :return: un tuple (ha,hs,oa,os) de ndarray de dimensions respectives
                (dh * n, dh * n, m * n, m * n).
        """
        # Le nombre d'exemples contenus dans train_x
        n = train_x.shape[0]
        # Transposer train_x afin d'obtenir un ndarray de dimension d * n
        train_x = np.transpose(train_x)
        # Donc W1 de dimension dh * d, train_x de dimension d * n  et
        # b1 à une dimension dh * n
        ha = np.dot(self.W1, train_x) + np.repeat(self.b1, n, axis=1)
        # Donc hs est dh * n
        hs = rect(ha)
        # W2 est m * dh   hs est dh * n  b1 doit donc être  m * n
        oa = np.dot(self.W2, hs) + np.repeat(self.b2, n, axis=1)
        # os est m * n
        os = softmax(oa)
        return ha, hs, oa, os

    def bprop(self, train_x, train_y):
        """
        Propagation arrière

        :param train_x: ndarray de dimension n * d
        :param train_y: ndarray de dimension n * 1

        :return:
        """
        # ha et hs de dimension dh * n et oa, os de dimension m * n
        ha, hs, oa, os = self.fprop(train_x)
        grad_oa = os - np.transpose(onehot(np.transpose(train_y),
                                                        self.m))
        grad_b2 = np.sum(grad_oa, axis=1).reshape(self.b2.shape)
        # Matrice m * n et n * dh donc grad_W2 est mat m * dh
        # La somme se fait automatiquement
        grad_w2 = np.dot(grad_oa, np.transpose(hs))
        # Régularisation
        grad_w2 += self.wd[2] + 2 * self.wd[3] * self.W2
        # Matrice dh * m et m * n donc grad_hs est dh * n
        grad_hs = np.dot(np.transpose(self.W2), grad_oa)
        # dh * n et dh * n (terme à terme) donc dh * n
        grad_ha = grad_hs * np.where(ha > 0, [1], [0])
        # dh * n
        grad_b1 = np.sum(np.array(grad_ha), axis=1).reshape(self.b1.shape)
        # Matrice dh * n et n * d
        grad_w1 = np.dot(grad_ha, train_x)
        # Régularisation
        grad_w1 += self.wd[0] + 2 * self.wd[1] * self.W1
        # dh * d, dh * n, m * dh, m * n
        return grad_w1, grad_b1, grad_w2, grad_b2

    def compute_grad(self, batch):
        """

        :param batch: un ndarray de dimension n * d + 1
        :return: un tuple (grad_w1, grad_b1, grad_w2, grad_b2) pour ce batch
        """
        # batch_x une matrice n * d
        batch_x = np.array(batch[:, :self.d])
        # batch_y une matrice n * 1
        batch_y = np.array(batch[:, -1]).reshape(batch.shape[0], 1)
        # Propagation arrière avec paramètres actuels
        return self.bprop(batch_x, batch_y)

    def compute_predictions(self, test_data):
        """

        :param test_data: un ndarray de dimension n * d

        :return: un vecteur de sorties de dimension n * 1
        """
        # Matrice n * m après transposition
        os = np.transpose(self.fprop(test_data)[3])
        # Retourner la classe ayant la plus haute probabilité pour la ligne
        # i qui correspond à l'exemple i (n exemples)
        return np.argmax(os, axis=1)

    def compute_loss(self, batch_x, batch_t):

        # Initialiser un vecteur-colonne de coût de dimension n * 1
        """

        :param batch_x: un ndarray de dimension n * d
        :param batch_t: un ndarray de dimension n * 1

        :return: cost: un ndarray de dimension n * 1 qui représente le coût cal-
                       culé pour chaque exemple.
        """
        cost = np.zeros(batch_t.shape)
        # Propagation avant du batch avec os de dimension m * n
        os = self.fprop(batch_x)[3]
        # Mise à jour des coûts pour chaque exemple
        for t, i in zip(np.nditer(batch_t), range(batch_t.shape[0])):
            cost[i][0] -= np.log(os[int(t)][i])

        return cost


def rect(x):
    """
    Rectifie terme à terme la liste ou ndarray

    :param x: une liste ou un ndarray
    :return: un ndarray de dimension même dimension que x mais dont les termes
    ont été rectifiés
    """
    return np.where(x > 0, [1], [0]) * np.array(x)


def softmax(x):
    """Calcule softmax pour un vecteur-colonne ou pour une matrice de vecteurs
    colonnes numériquement stable """

    # Si c'est un unique vecteur-colonne
    if x.shape[1] == 1:

        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum()

    # Sinon appeler softmax colonne par colonne sur la matrice x
    else:

        res = []

        for col in range(x.shape[1]):
            # On reshape afin de pouvoir passer un vecteur colonne à softmax
            vec = softmax(np.array(x[:, col]).reshape(x.shape[0], 1))
            # On obtient en retour un vecteur colonne que l'on ajoute
            # comme vecteur-ligne dans la liste res
            res.append(vec[:, -1])

        # On transpose la matrice res afin de transformer la matrice de
        # vecteurs-lignes en matrice de vecteurs-colonnes
        return np.transpose(np.array(res))


def onehot(pos, length):
    """

    :param pos: ndarray de int de dimension n * 1 représentant les positions
                à encoder dans onehot
    :param length: int représentant la largeur en nombre de colonnes de l'enco-
                   dage onehot.

    :return: un ndarray de dimension len(pos) * length représentant
             une matrice d'encodage onehot pour toutes les positions du
             paramètre pos
    """
    # Récupérer les positions sous forme de liste
    pos = pos[0, :]
    # Initialiser une matrice c de 0
    c = np.zeros((pos.shape[0], length))
    # Pour chaque ligne i, on ajoute 1 à la position désigné par pos
    # puis on incrémente i et on avance dans la liste pose
    for i, y in zip(range(len(pos)), pos):
        c[i, int(y)] += 1
    return c
