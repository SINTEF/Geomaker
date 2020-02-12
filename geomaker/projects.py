class Project:

    def __init__(self, key, name, coords):
        self.key = key
        self.name = name
        self.coords = coords

    def __lt__(self, other):
        return self.key < other.key

    def __str__(self):
        return self.key


class DigitalHeightModel(Project):

    def __init__(self, key, name):
        super().__init__(key, name, 'utm33n')


class TiledImageModel(Project):

    def __init__(self, key, name):
        super().__init__(key, name)
