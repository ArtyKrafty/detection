class Pipeline(object):

    """
    
    Класс-генератор. Основа всех классов.
    ___or___ позволяет перебирать классы в порядке, 
    установленном в пайплайне. Генератор - 
    сохраняет данные и передает на следующий шаг
    
    """

    def __init__(self, source=None):
        self.source = source

    def __iter__(self):
        return self.generator()

    def generator(self):


        while self.has_next():
            try:
                data = next(self.source) if self.source else {}
                if self.filter(data):
                    yield self.map(data)
            except StopIteration:
                return

    def __or__(self, other):


        if other is not None:
            other.source = self.generator()
            return other
        else:
            return self

    def filter(self, data):


        return True

    def map(self, data):


        return data

    def has_next(self):


        return True