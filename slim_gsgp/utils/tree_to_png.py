# MIT License
#
# Copyright (c) 2024 DALabNOVA
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
"""
Tree visualization to PNG using matplotlib and networkx.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import networkx as nx
import re
from typing import Tuple, Dict, Any
import os


def clean_node_name(node):
    """
    Clean node names by removing np.str_ wrappers and quotes.
    
    Parameters
    ----------
    node : str or tuple
        Node to clean
        
    Returns
    -------
    str
        Cleaned node name
    """
    if isinstance(node, tuple):
        return str(node)
    
    node_str = str(node)
    # Remove np.str_() wrapper
    if "np.str_(" in node_str:
        node_str = re.sub(r"np\.str_\('([^']+)'\)", r"\1", node_str)
    
    # Remove quotes
    node_str = node_str.strip("'\"")
    
    # Clean up constant names
    if node_str.startswith("constant_"):
        node_str = node_str.replace("constant_", "")
    
    # Convert operator names to mathematical symbols
    operator_symbols = {
        'add': '+',
        'subtract': '-',
        'multiply': '*',
        'divide': '/'
    }
    
    if node_str in operator_symbols:
        node_str = operator_symbols[node_str]
    
    return node_str


def tree_to_graph(tree_structure):
    """
    Convert tree structure to NetworkX graph.
    
    Parameters
    ----------
    tree_structure : tuple or str
        Tree structure to convert
        
    Returns
    -------
    nx.DiGraph
        NetworkX directed graph representing the tree
    dict
        Node labels mapping
    dict
        Node colors mapping
    """
    G = nx.DiGraph()
    labels = {}
    colors = {}
    node_counter = [0]  # Use list to make it mutable in nested function
    
    def add_nodes_recursive(structure, parent=None):
        """Recursively add nodes to the graph."""
        current_id = node_counter[0]
        node_counter[0] += 1
        
        if isinstance(structure, tuple) and len(structure) >= 3:
            # Internal node (operator)
            operator = clean_node_name(structure[0])
            labels[current_id] = operator
            colors[current_id] = 'lightblue'  # Operators in light blue
            G.add_node(current_id)
            
            if parent is not None:
                G.add_edge(parent, current_id)
            
            # Add children
            for i in range(1, len(structure)):
                add_nodes_recursive(structure[i], current_id)
                
        else:
            # Leaf node (terminal or constant)
            node_name = clean_node_name(structure)
            labels[current_id] = node_name
            
            # Color coding
            if node_name.startswith('x'):
                colors[current_id] = 'lightgreen'  # Variables in light green
            elif node_name.replace('.', '').replace('-', '').isdigit():
                colors[current_id] = 'lightcoral'  # Constants in light coral
            else:
                colors[current_id] = 'lightyellow'  # Other terminals in light yellow
                
            G.add_node(current_id)
            
            if parent is not None:
                G.add_edge(parent, current_id)
    
    add_nodes_recursive(tree_structure)
    return G, labels, colors


def save_tree_as_png(tree_structure, filename="tree_visualization.png", figsize=(16, 12), dpi=300):
    """
    Save tree structure as PNG image.
    
    Parameters
    ----------
    tree_structure : tuple or str
        Tree structure to visualize
    filename : str, optional
        Output filename
    figsize : tuple, optional
        Figure size (width, height) in inches
    dpi : int, optional
        DPI for the output image
        
    Returns
    -------
    str
        Path to the saved image
    """
    # Convert tree to graph
    G, labels, colors = tree_to_graph(tree_structure)
    
    # Create figure
    fig, ax = plt.subplots(1, 1, figsize=figsize, dpi=dpi)
    
    # Use hierarchical layout
    try:
        # Try to use graphviz layout for better tree visualization
        pos = nx.nx_agraph.graphviz_layout(G, prog='dot')
    except:
        # Fallback to spring layout if graphviz is not available
        pos = nx.spring_layout(G, k=3, iterations=50)
    
    # Get node colors list in the correct order
    node_colors = [colors[node] for node in G.nodes()]
    
    # Draw the graph
    nx.draw(G, pos, 
            labels=labels,
            node_color=node_colors,
            node_size=2000,
            font_size=10,
            font_weight='bold',
            font_color='black',
            edge_color='gray',
            arrows=True,
            arrowsize=20,
            arrowstyle='->',
            ax=ax)
    
    # Add title
    ax.set_title("SLIM GSGP Tree Structure", fontsize=16, fontweight='bold', pad=20)
    
    # Add legend
    legend_elements = [
        patches.Patch(color='lightblue', label='Operators'),
        patches.Patch(color='lightgreen', label='Variables'),
        patches.Patch(color='lightcoral', label='Constants'),
        patches.Patch(color='lightyellow', label='Other')
    ]
    ax.legend(handles=legend_elements, loc='upper right')
    
    # Remove axes
    ax.set_axis_off()
    
    # Adjust layout and save
    plt.tight_layout()
    plt.savefig(filename, dpi=dpi, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    plt.close()
    
    # Get absolute path
    abs_path = os.path.abspath(filename)
    print(f"✅ Tree visualization saved to: {abs_path}")
    return abs_path


def save_tree_as_png_simple(tree_structure, filename="tree_simple.png", figsize=(24, 18)):
    """
    Save tree as PNG using a simpler matplotlib-only approach.
    
    Parameters
    ----------
    tree_structure : tuple or str
        Tree structure to visualize
    filename : str, optional
        Output filename
    figsize : tuple, optional
        Figure size
        
    Returns
    -------
    str
        Path to saved image
    """
    fig, ax = plt.subplots(1, 1, figsize=figsize, dpi=200)
    
    # Manually draw tree structure
    def get_tree_dimensions(structure):
        """Calculate tree dimensions for positioning."""
        if not isinstance(structure, tuple) or len(structure) < 3:
            return 1, 1  # width, height
        
        heights = []
        total_width = 0
        
        for i in range(1, len(structure)):
            child_width, child_height = get_tree_dimensions(structure[i])
            total_width += child_width
            heights.append(child_height)
        
        return max(total_width, 1), max(heights) + 1 if heights else 1
    
    def draw_tree_recursive(structure, x, y, width, level=0):
        """Recursively draw tree nodes."""
        if not isinstance(structure, tuple) or len(structure) < 3:
            # Leaf node
            node_name = clean_node_name(structure)
            
            # Choose color based on node type
            if node_name.startswith('x'):
                color = 'lightgreen'
            elif node_name.replace('.', '').replace('-', '').isdigit():
                color = 'lightcoral'
            else:
                color = 'lightyellow'
            
            # Draw node - make it larger for better text visibility
            circle = plt.Circle((x, y), 1.0, color=color, ec='black', linewidth=2)
            ax.add_patch(circle)
            ax.text(x, y, node_name, ha='center', va='center', fontsize=5, fontweight='bold')
            return x, y
        
        # Internal node (operator)
        operator = clean_node_name(structure[0])
        
        # Draw operator node - make it larger for better text visibility
        circle = plt.Circle((x, y), 1.0, color='lightblue', ec='black', linewidth=2)
        ax.add_patch(circle)
        ax.text(x, y, operator, ha='center', va='center', fontsize=5, fontweight='bold')
        
        # Draw children
        num_children = len(structure) - 1
        if num_children > 0:
            child_spacing = width / num_children
            start_x = x - width/2 + child_spacing/2
            
            for i in range(1, len(structure)):
                child_x = start_x + (i-1) * child_spacing
                child_y = y - 4.0  # Increased vertical separation
                
                # Draw edge - adjust for larger nodes and spacing
                ax.plot([x, child_x], [y-0.9, child_y+0.9], 'k-', linewidth=2)
                
                # Draw child with more horizontal spacing
                child_width = width / num_children * 1.5  # Increased horizontal spacing
                draw_tree_recursive(structure[i], child_x, child_y, child_width, level+1)
        
        return x, y
    
    # Calculate dimensions and draw - further increase spacing
    tree_width, tree_height = get_tree_dimensions(tree_structure)
    draw_tree_recursive(tree_structure, 0, tree_height * 4, tree_width * 6)
    
    # Set limits and styling - adjust for increased spacing
    ax.set_xlim(-tree_width*6, tree_width*6)
    ax.set_ylim(-3, tree_height*4 + 3)
    ax.set_aspect('equal')
    ax.set_axis_off()
    ax.set_title("SLIM GSGP Tree Visualization", fontsize=16, fontweight='bold', pad=20)
    
    # Save
    plt.tight_layout()
    plt.savefig(filename, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close()
    
    abs_path = os.path.abspath(filename)
    print(f"✅ Tree visualization saved to: {abs_path}")
    return abs_path