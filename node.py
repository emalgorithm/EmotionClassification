class Node(object):
    def __init__(self, kids = [], op = -1, c = -1, confidence = 0):
        # 1x2 array containing left and right children
        self.kids = kids
        # number, index of the attribute the node is splitting on
        self.op = op
        # number, only for leaf node: index of predicted class for this path
        self.c = c

        self.confidence = confidence

    def __repr__(self, level = 0):
        ret = ""
        if self.c != -1:
            ret = "\t" * level + str(self.op) + ", " + str(self.c) + "\n"
        else:
            ret = "\t" * level + str(self.op) + "\n"
        for kid in self.kids:
            ret += kid.__repr__(level + 1)
        return ret

    