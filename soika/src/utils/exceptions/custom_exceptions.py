class BaseError(Exception):
    def __init__(self, *args):
        if args:
            self.message = args[0]
        else:
            self.message = None

    def __str__(self):
        if self.message:
            return "{0}, {1} ".format(self.__class__.__name__, self.message)
        else:
            return self.__class__.__name__


class InvalidInputError(BaseError):
    pass


class ClassifierInitializationError(BaseError):
    pass


class DataError(BaseError):
    pass

class ConectionError(BaseError):
    pass

class ClassificationError(BaseError):
    pass