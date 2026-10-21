class Event:
    """
    Base class for events.
    """

    def __init__(self, t):
        self.t = t

    def __str__(self):
        raise NotImplementedError("Subclasses must implement __str__ method")

    def to_bytes(self) -> bytes:
        """
        Convert the event to a unique byte representation for hashing.
        """
        return self.__str__().encode('utf-8')

    def apply(self, scenario):
        raise NotImplementedError
