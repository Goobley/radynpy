class Val:
    def __init__(self, val):
        self.val = val

    def __repr__(self):
        return 'Val(%s)' % self.val.__repr__()

class Array:
    def __init__(self, shape):
        self.shape = shape

    def __repr__(self):
        return 'Array%s' % self.shape.__repr__()

    def idl_repr(self):
        s = '['
        for i, idx in enumerate(self.shape):
            assert(type(idx) == str)
            s += idx
            if i != len(self.shape)-1:
                s +=','
        s += ']'
        return s

class Unknown:
    def __init__(self):
        pass

    def __repr__(self):
        return 'Unknown'

