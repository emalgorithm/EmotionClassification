class Node:
    # label (attribute) the node is testing
    # op
    # left / right subtrees
    # kids
    # label for the leaf nodes
    # classification
    # confidence

    def __init__(self, op, kids, classification = None, confidence = 100):
        self.op = op
        self.kids = [None] * 2
        self.classification = classification
        self.confidence = confidence