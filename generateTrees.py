import random

class Node:
    def __init__(self, data, right = None, left = None):
        self.data = data
        self.right = right
        self.left = left
        self.id = id(self)
        

def serialize( root):
    def post_order(root):
        if root:
            post_order(root.left)
            post_order(root.right)
            ret[0] += str(root.data)+';'
                
        else:
            ret[0] += '99;'           

    ret = ['']
    post_order(root)
    return ret[0][:-1]  # remove last ,

def deserialize(data):
    if  not data:
        return 
    nodes = data.split(';')  
    def post_order(nodes):
        if nodes[-1] == '99':
            nodes.pop()
            return None
        node = nodes.pop()
        
        data = int(node)
        root = Node(data)
        root.right = post_order(nodes)
        root.left = post_order(nodes)
        
        return root    
    return post_order(nodes)  



def generate_random_tree(depth):
    if depth <= 1 or random.random() < 0.2:
        return Node(random.randint(1, 98))
    
    root = Node(random.randint(1, 98))
    root.left = generate_random_tree(depth - 1)
    root.right = generate_random_tree(depth - 1)

    return root

def print_tree(root, level=0, prefix="Root: "):
    if root is not None:
        print(" " * (level * 4) + prefix + str(root.data))
        if root.left is not None or root.right is not None:
            print_tree(root.left, level + 1, "L--- ")
            print_tree(root.right, level + 1, "R--- ")

# Generate and print 10 random tree
if __name__=="__main__":
    for i in range(1, 11):
        tree = generate_random_tree(4)  # You can adjust the depth as needed
        print(f"\nRandom Tree {i}:\n")
        #print_tree(tree)
        serial = serialize(tree) 
        print(serial)
        lista = serial.split(';')
        print("list", lista)
        file = open("trees/arbol" + str(i) + ".dat", "w")
        file.write(serial)
        file.close() 