import random

class Node:
    def __init__(self, data, right = None, left = None):
        self.data = data
        self.right = right
        self.left = left
        

def serialize( root):
    def post_order(root):
        if root:
            post_order(root.left)
            post_order(root.right)
            ret[0] += str(root.data)+';'
                
        else:
            ret[0] += '#;'           

    ret = ['']
    post_order(root)
    return ret[0][:-1]  # remove last ,

def deserialize(data):
    if  not data:
        return 
    nodes = data.split(';')  
    def post_order(nodes):
        if nodes[-1] == '#':
            nodes.pop()
            return None
        node = nodes.pop()
        
        data = int(node)
        root = Node(data)
        root.right = post_order(nodes)
        root.left = post_order(nodes)
        
        return root    
    return post_order(nodes)  

class TreeNode:
    def __init__(self, data):
        self.data = data
        self.left = None
        self.right = None

def generate_random_tree(depth):
    if depth == 0 or random.random() < 0.2:
        return None
    root = TreeNode(random.randint(1, 100))
    root.left = generate_random_tree(depth - 1)
    root.right = generate_random_tree(depth - 1)
    return root

def print_tree(root, level=0, prefix="Root: "):
    if root is not None:
        print(" " * (level * 4) + prefix + str(root.data))
        if root.left is not None or root.right is not None:
            print_tree(root.left, level + 1, "L--- ")
            print_tree(root.right, level + 1, "R--- ")

# Generate and print 10 random trees
for i in range(1, 11):
    tree = generate_random_tree(4)  # You can adjust the depth as needed
    print(f"\nRandom Tree {i}:\n")
    #print_tree(tree)
    print(serialize(tree))
    l = list(serialize(tree))
    print("list", [a for a in l if a != ";"])