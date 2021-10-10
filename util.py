""" Module containing general utility functions """

def all_subclasses(c):
    """ Return a set of all the subclasses of the class c using recursion """
    ac = set.union(
        set(c.__subclasses__()),
        [ssc for sc in c.__subclasses__() for ssc in all_subclasses(sc)],
    )
    return ac
