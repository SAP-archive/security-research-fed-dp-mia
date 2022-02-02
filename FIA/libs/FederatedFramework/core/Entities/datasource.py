import collections
import numpy as np
from sklearn.model_selection import train_test_split


class Datasource():
    """
    Define the data source for the federated parties
    """

    def __init__(self, alternative_dataset=None):
        self.x = np.array([])
        self.y = np.array([])

    def get_data(self):
        return self.x, self.y

    def get_data_for_class(self, position, num_clients, seed):
        """
        Return disjunct (X,y) sets for each class from classes
        :param classes:
        :return:
        """
        np.random.seed(seed) # set seed, so for every process its the same outcome
        # First give the Aggregator some data
        proportion = 5  / 7
        indices = np.random.choice(self.x.shape[0], size=self.x.shape[0], replace=False)
        do_indices, ag_indices = train_test_split(indices, test_size=proportion)
        if position == 0:
            return ag_indices

        amount_samples = []
        num_clients = num_clients - 1 # remove the server
        position = position - 1       # remove the server
        # Second execute round robin
        y = self.y[do_indices]
        all_classes = list(set(y)) # get unique classes
        class_distribution = list(collections.Counter(y)) # count each sample for each class
        sorted_class_by_distribution = list(sorted(all_classes, key=lambda y: -class_distribution[y])) # sort the unique classes by its number of occurences

        do = {} # all data owners data
        cursor = 0 # the current data owner
        while len(sorted_class_by_distribution) > 0: # as long as we have classes, distribute them
            c = sorted_class_by_distribution.pop()  # remove the top class
            amount_samples = [sum([class_distribution[x] for x in do[i]]) for i in range(0, num_clients) if i in do]
            # If the data owner has not yet given a class or it has fewer classes than other data owners
            if not cursor in do or amount_samples[cursor] <= min(amount_samples):
                if not cursor in do:
                    do[cursor] = [c]
                else:
                    do[cursor].append(c) # give them a class

            cursor = (cursor + 1) % num_clients # rotate around the data owners
        assert set(all_classes).difference(set(np.concatenate(list(do[x] for x in range(0, num_clients))))) == set(), "all classes must be distributed"
        print("cursor", position, "amount", amount_samples[position], "classes", do[position])
        return [x for x in do_indices if self.y[x] in do[position]]


    def disjoint_dataset_indices(self, position, num_clients, seed):
        """
        Get disjoint data sets for each party.
        It is important to set the seed and the amount of clients for each party to the same value
        otherwise it is not guaranteed that the sets are disjoint
        """
        np.random.seed(seed)
        indices = np.random.choice(self.x.shape[0], size=self.x.shape[0], replace=False)
        batch_size = self.x.shape[0] // num_clients
        return indices[position * batch_size:(position + 1) * batch_size]
