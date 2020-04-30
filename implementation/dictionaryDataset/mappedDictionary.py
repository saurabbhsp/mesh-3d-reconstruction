"""
from dictionaryDataset.core import PartialDictionarySet


class MappedDictionary(PartialDictionarySet):

    def __init__(self, base_dataset, map_fn):
        self._map_fn = map_fn
        super(MappedDictionary, self).__init__(base_dataset)

    def __contains__(self, key):
        return key in self._base

    def __get_item__(self, key):
        return self._map_fn(self._base[key])

    def __len__(self):
        return len(self._base)
"""
