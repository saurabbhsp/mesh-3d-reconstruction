import pickle


def write_list(path, list):
    with open(path, 'wb') as f:
        pickle.dump(list, f)


def read_list(path):
    list = None
    with open(path, 'rb') as f:
        list = pickle.load(f)
    return list
