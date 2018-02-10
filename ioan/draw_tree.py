from node import Node
import turtle
    
# def deserialize(string):
#     if string == '{}':
#         return None
#     nodes = [None if val == 'null' else TreeNode(int(val))
#              for val in string.strip('[]{}').split(',')]
#     kids = nodes[::-1]
#     root = kids.pop()
#     for node in nodes:
#         if node:
#             if kids: node.left  = kids.pop()
#             if kids: node.right = kids.pop()
#     return root

""" Code for drawing Binary Trees using turtle graphics.  

This code is likely to be somewhat fragile, as it relies on
undocumented internal attributes of the tree class.  In particular
it requires that:

The tree class have a _root instance variable that contains a node and
that the node objects have attributes named :

  op
  left 
  right

Author: Nathan Sprague, Nov. 2011
"""
def height(root):
        return 1 + max(height(root.kids[0]), height(root.kids[1])) if root else -1

#drawing constants:
HEIGHT_LEVEL = 80
WIDTH_CHAR = 10


def visualise(tree):
    """ Draw a binary search tree using turtle graphics. """

    screenWidth = _pixelWidth(tree)
    if tree.kids[1] is not None:
        rightSubtreeWidth = _pixelWidth(tree.kids[1])
    else:
        rightSubtreeWidth = 0

    turtle.setup(screenWidth + 50, height(tree) * HEIGHT_LEVEL + 50)
    turtle.screensize(screenWidth, height(tree) * HEIGHT_LEVEL)

    turtle.pu()

    turtle.goto(.5 * screenWidth - (rightSubtreeWidth + 20),  
                (height(tree) - 1) * HEIGHT_LEVEL / 2)
    turtle.pd()

    turtle.speed(0)
    turtle.hideturtle()
    _drawTreeRec(tree)
    turtle.exitonclick()

def _drawTreeRec(subtree):
    # Recursive helper method for drawTree. 
    if subtree != None:
        turtle.color("black")
        turtle.write("AU" + str(subtree.op) if subtree.op else str(subtree.classification))
        turtle.color("blue")
        turtle.dot()
        cur_pos = turtle.position()

        if subtree.kids[0] != None:
            turtle.goto(cur_pos[0] - _pixelWidthRight(subtree.kids[0]), 
                        cur_pos[1] - HEIGHT_LEVEL)
            _drawTreeRec(subtree.kids[0])
            turtle.goto(cur_pos)

        if subtree.kids[1] != None:
            turtle.goto(cur_pos[0] + _pixelWidthLeft(subtree.kids[1]), 
                        cur_pos[1] - HEIGHT_LEVEL)
            _drawTreeRec(subtree.kids[1])
            turtle.goto(cur_pos)


def _pixelWidth(subtree):
    # Recursively determine the number of pixels needed to represent
    # an entire subtree.
    if subtree == None:
        return 0
    else:
        return (len(str(subtree.op) * WIDTH_CHAR) +
                _pixelWidth(subtree.kids[0]) +
                _pixelWidth(subtree.kids[1]))

def _pixelWidthLeft(subtree):
    # Determine the number of pixels needed to represent this node and
    # its entire left subtree.
    if subtree == None:
        return 0
    else:
        return (len(str(subtree.op) * WIDTH_CHAR) +
                _pixelWidth(subtree.kids[0]))

def _pixelWidthRight(subtree):
    # Determine the number of pixels needed to represent this node and
    # its entire right subtree.
    if subtree == None:
        return 0
    else:
        return (len(str(subtree.op) * WIDTH_CHAR) +
                _pixelWidth(subtree.kids[1]))

# if __name__ == '__main__':
#     drawTree(deserialize('[1,2,3,null,null,4,null,null,5]'))
