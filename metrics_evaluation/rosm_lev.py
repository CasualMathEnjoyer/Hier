import numpy as np
class LevenshteinDistance:
    '''
    A class representing an enhanced levenshtein distance.
    In this class, a number of permutation have a lower cost.
    '''

    def __init__(self):
        self.reduceCost = set()
        pass

    def addPermutation(self, c1, c2):
        '''
        Allows permutation c1/c2 with a cost of 1.
        '''
        pass

    def compute(self, text1, text2, remove_spaces=True):
        '''
        Compute this distance between two strings.
        - remove_spaces : use if space are separators (which is the case here)
        '''
        if remove_spaces:
            text1 = text1.replace(" ", "")
            text2 = text2.replace(" ", "")
        long1 = len(text1)
        long2 = len(text2)
        tab = np.zeros((long1 + 1, long2 + 1))
        # fill first range :
        tab[0] = [v for v in range(0, long2 + 1)]
        for i in range(1, long1 + 1):
            tab[i][0] = i
            for j in range(1, long2 + 1):
                addDist = tab[i][j - 1] + 1
                removeDist = tab[i - 1][j] + 1
                valDiag = tab[i - 1][j - 1]
                if text1[i - 1] == text2[j - 1]:
                    substDist = valDiag
                else:
                    substDist = valDiag + 1
                tab[i][j] = min((addDist, removeDist, substDist))
        return tab[long1][long2]

    def fullDist(self, fname1, fname2, enc="utf-8"):
        '''
        Display (on stdout) detailled statistics about the distances between two files
        Column-separated output will be:
        - line number
        - length of fname1 entry
        - length of fname2 entry
        - levenshtein distance between the two entries.
        '''
        with open(fname1, mode="r", encoding=enc) as f1:
            with open(fname2, mode="r", encoding=enc) as f2:
                for line_number, (l1, l2) in enumerate(zip(f1, f2)):
                    l1 = l1.strip()
                    l2 = l2.strip()
                    l1_length = len(l1.split(' '))
                    l2_length = len(l2.split(' '))
                    # REMOVE SPACES TO COMPUTE THE CORRECT DISTANCE...
                    distance = self.compute(l1, l2)
                    # print("{}\t{}\t{:7.3f}\t{:7.3f}\t{}\t{:7.3f}\t{:7.3f}".format(key, *value))
                    print("{}:{}:{}:{:7.3f}".format(line_number, l1_length, l2_length, distance))

    def meanDist(self, fname1, fname2, enc="utf-8"):
        '''
        - fname1 : le fichier attendu
        - fname2 : le fichier obtenu
        Retourne un dictionnaire, dont les clefs sont les longueur des phrases dans fname1,
        et les valeurs des n-uplets dont les champs sont :
        - n : le nombre d'occurrences de longueur
        - err : la somme des distances levenshtein pour cette longueur
        - m : la moyenne pour cette longueur (0 par convention si n = 0), divisée par la longueur
        - n_tot : le nombre cumulé d'occurrences pour n=0..n
        - err_tot : l'erreur cumulée, CHAQUE ERREUR EST RAPPORTÉE À SA LONGUEUR
        - m_tot : la moyenne cumulée (sans doute la plus significative) CHAQUE ERREUR EST DIVISEE PAR SON N

        La ligne pour le n maximal donne donc l'erreur sur le corpus entier.

        '''
        res = dict()
        with open(fname1, mode="r", encoding=enc) as f1:
            with open(fname2, mode="r", encoding=enc) as f2:
                for l1, l2 in zip(f1, f2):
                    l1 = l1.strip()
                    l2 = l2.strip()
                    l1_length = len(l1.split(' '))
                    if l1_length in res.keys():
                        n, err, _, _, _, _ = res[l1_length]
                    else:
                        n = 0
                        err = 0
                    n += 1
                    err += self.compute(l1, l2)
                    res[l1_length] = (n, err, 0, 0, 0, 0)
        n_tot = 0
        err_tot = 0
        for current_length in sorted(res.keys()):
            n, err, _, _, _, _ = res[current_length]
            n_tot += n * current_length
            err_tot += err
            moyenne = err / n / current_length
            moyenne_tot = err_tot / n_tot
            res[current_length] = (n, err, moyenne, n_tot, err_tot, moyenne_tot)
        return res
