import csv
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle


def get_accuracy(network, dataset_x, dataset_y):
    c = 0
    for i in range(dataset_y.size):
        o = 1 if network.propagate_forward(dataset_x[i])[0] >= 0 else -1

        if dataset_y[i] == o:
            c += 1
    return (100.0 * c / dataset_y.size)


def get_accuracy_threeclass(network, dataset_x, dataset_y):
    c = 0
    for i in range(len(dataset_y)):
        o = np.argmax(network.propagate_forward(dataset_x[i]))

        if np.argmax(dataset_y[i]) == o:
            c += 1
    return (100.0 * c / len(dataset_y))


def learn(network, ds_x, ds_y, two_class, epochs, lrate, momentum):
    train_ds_x, test_ds_x, train_ds_y, test_ds_y = train_test_split(ds_x, ds_y, test_size=0.3)

    # train_ds_x = ds_x[:int(0.7*len(ds_x))]
    # train_ds_y = ds_y[:int(0.7*len(ds_y))]
    # test_ds_x = ds_x[int(0.7*len(ds_x)):]
    # test_ds_y = ds_y[int(0.7*len(ds_y)):]

    # Training
    for e in range(epochs):
        for i in range(len(train_ds_y)):
            network.propagate_forward(train_ds_x[i])
            network.propagate_backward(train_ds_y[i], lrate, momentum)
    if two_class == True:
        print('Train accuracy in end of ' + str(e + 1) + ' epochs = %' + '%.2f' % get_accuracy(network, train_ds_x,
                                                                                               train_ds_y))

        # Test
        print('Test accuracy = %' + '%.2f' % get_accuracy(network, test_ds_x, test_ds_y))

    elif two_class == False:
        print('Train accuracy in end of ' + str(e + 1) + ' epochs = %' + '%.2f' % get_accuracy_threeclass(network,
                                                                                                          train_ds_x,
                                                                                                          train_ds_y))

        # Test
        print('Test accuracy = %' + '%.2f' % get_accuracy_threeclass(network, test_ds_x, test_ds_y))


def indices_to_one_hot(data, nb_classes):
    """Convert an iterable of indices to one-hot encoded labels."""
    targets = np.array(data).reshape(-1)
    return np.eye(nb_classes)[targets]


def ds_read(ds_name: str):
    x = np.array([])
    y = np.array([])
    with open('../datasets/' + ds_name + '.dat') as f:
        if ds_name != 'Hayes-roth':
            reader = csv.reader(f, delimiter=",")

            if ds_name == 'Banana':
                for i in range(7):
                    next(reader)
                for line in reader:
                    x = np.append(x, [float(line[0]), float(line[1])])
                    y = np.append(y, [float(line[2])])
                x = x.reshape(-1, 2)

            elif ds_name == 'Haberman':
                for i in range(8):
                    next(reader)
                for line in reader:
                    x = np.append(x, [int(line[0]), int(line[1]), int(line[2])])
                    y = np.append(y, [1 if (line[3] == ' positive') else -1])
                x = x.reshape(-1, 3)

            elif ds_name == 'Titanic':
                for i in range(8):
                    next(reader)
                for line in reader:
                    x = np.append(x, [float(line[0]), float(line[1]), float(line[2])])
                    y = np.append(y, [float(line[3])])
                x = x.reshape(-1, 3)

            elif ds_name == 'Balance':
                for i in range(9):
                    next(reader)
                for line in reader:
                    x = np.append(x, [float(line[i]) for i in range(4)])

                    if line[4] == ' L':
                        l4 = 0
                    elif line[4] == ' R':
                        l4 = 1
                    elif line[4] == ' B':
                        l4 = 2
                    y = np.append(y, [l4])
                x = x.reshape(-1, 4)

                y = np.array(y, dtype=int)
                y = indices_to_one_hot(list(y), nb_classes=3)

            elif ds_name == 'Newthyroid':
                for i in range(10):
                    next(reader)
                for line in reader:
                    x = np.append(x, [float(line[i]) for i in range(5)])
                    y = np.append(y, [float(line[5]) - 1])
                x = x.reshape(-1, 5)

                y = np.array(y, dtype=int)
                y = indices_to_one_hot(list(y), nb_classes=3)

            elif ds_name == 'Wine':
                for i in range(18):
                    next(reader)
                for line in reader:
                    x = np.append(x, [float(line[i]) for i in range(13)])
                    y = np.append(y, [float(line[13]) - 1])
                x = x.reshape(-1, 13)

                y = np.array(y, dtype=int)
                y = indices_to_one_hot(list(y), nb_classes=3)

        elif ds_name == 'Hayes-roth':
            content = f.readlines()

            for c in content[9:len(content) - 1]:
                x = np.append(x, [int(c[3 * i]) for i in range(4)])
                y = np.append(y, [int(c[12]) - 1])
            x = x.reshape(-1, 4)

            y = np.array(y, dtype=int)
            y = indices_to_one_hot(list(y), nb_classes=3)

    return shuffle(x, y)
