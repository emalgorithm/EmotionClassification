from node import Node
import turtle
    
def tree_height(root):
  return 1 + max(tree_height(root.kids[0]), tree_height(root.kids[1])) if root.kids else -1

# drawing constants
height = 80
width = 10

# draws a binary search tree using turtle graphics.
def visualise(tree):
  screenWidth = _pixelWidth(tree.root)
  if tree.root.kids[1] is not None:
    rightSubtreeWidth = _pixelWidth(tree.root.kids[1])
  else:
    rightSubtreeWidth = 0
  print(tree_height(tree.root))
  turtle.setup(screenWidth + 50, tree_height(tree.root) * height + 50)
  turtle.screensize(screenWidth, tree_height(tree.root) * height)

  turtle.pu()
  turtle.goto(.5 * screenWidth - (rightSubtreeWidth + 20),  (tree_height(tree.root) - 1) * height / 2)
  turtle.pd()

  turtle.speed(0)
  turtle.hideturtle()
  drawTreeRec(tree.root)
  turtle.exitonclick()

def drawTreeRec(subtree):
    # Recursive helper method for drawTree. 
    if subtree.kids:
        turtle.color("black")
        turtle.write("AU" + str(subtree.op) if subtree.op != -1 else str(subtree.c))
        turtle.color("blue")
        turtle.dot()
        cur_pos = turtle.position()

        if subtree.kids[0].kids:
            turtle.goto(cur_pos[0] - _pixelWidthRight(subtree.kids[0]), 
                        cur_pos[1] - height)
            drawTreeRec(subtree.kids[0])
            turtle.goto(cur_pos)

        if subtree.kids[1].kids:
            turtle.goto(cur_pos[0] + _pixelWidthLeft(subtree.kids[1]), 
                        cur_pos[1] - height)
            drawTreeRec(subtree.kids[1])
            turtle.goto(cur_pos)


def _pixelWidth(subtree):
    # Recursively determine the number of pixels needed to represent
    # an entire subtree.
    if subtree.kids == []:
        return 0
    else:
        return (len(str(subtree.op) * width) +
                _pixelWidth(subtree.kids[0]) +
                _pixelWidth(subtree.kids[1]))

def _pixelWidthLeft(subtree):
    # Determine the number of pixels needed to represent this node and
    # its entire left subtree.
    if subtree.kids == []:
        return 0
    else:
        return (len(str(subtree.op) * width) +
                _pixelWidth(subtree.kids[0]))

def _pixelWidthRight(subtree):
    # Determine the number of pixels needed to represent this node and
    # its entire right subtree.
    if subtree.kids == []:
        return 0
    else:
        return (len(str(subtree.op) * width) +
                _pixelWidth(subtree.kids[1]))