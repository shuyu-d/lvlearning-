# === NODE PROPERTIES
import numpy as np

def parents(B, node):
    pa = np.where( B[:, node] != 0)[0]
    return set(pa)

def children(B, node):
    ch = np.where( B[node, :] != 0)[0]
    return set(ch)

def pc_set(B, node):
    return set.union(parents(B,node), children(B,node))

def parents_of(B, nodes):
    """
    Return all nodes that are parents of the node or set of nodes ``nodes``.

    Parameters
    ----------
    nodes
        A node or set of nodes.

    See Also
    --------
    children_of, neighbors_of, markov_blanket_of

    Examples
    --------
    >>> import causaldag as cd
    >>> g = cd.DAG(arcs={(1, 2), (2, 3)})
    >>> g.parents_of(2)
    {1}
    >>> g.parents_of({2, 3})
    {1, 2}
    """
    return set.union(*(parents(B,n) for n in nodes))


def children_of(B, nodes):
    """
    Return all nodes that are children of the node or set of nodes ``nodes``.

    Parameters
    ----------
    nodes
        A node or set of nodes.

    See Also
    --------
    parents_of, neighbors_of, markov_blanket_of

    Examples
    --------
    >>> import causaldag as cd
    >>> g = cd.DAG(arcs={(1, 2), (2, 3)})
    >>> g.children_of(1)
    {2}
    >>> g.children_of({1, 2})
    {2, 3}
    """
    return set.union(*(children(B,n) for n in nodes))


def spouses_of(B, node):
    """
    Return the set of spouses of ``node``, i.e., parents of the children of ``node``.

    Parameters
    ----------
    node:
        Node whose Markov blanket to return.

    See Also
    --------
    parents_of, children_of, neighbors_of

    Returns
    -------
    set:
        the spouses of ``node``.
    Example
    -------
    >>>
    >>>
    >>>
    {0, 2, 3}
    """
    ch_ = children(B, node)
    # print('chidren set is', ch_)
    if ch_:
        parents_of_children = set.union(*(parents(B,c) for c in ch_))
    else:
        parents_of_children = set()
    return parents_of_children - {node}

def markov_blanket_of(B, node):
    """
    Return the Markov blanket of ``node``, i.e., the parents of the node, its children, and the parents of its children.

    Parameters
    ----------
    node:
        Node whose Markov blanket to return.

    See Also
    --------
    parents_of, children_of, neighbors_of

    Returns
    -------
    set:
        the Markov blanket of ``node``.

    Example
    -------
    >>> import causaldag as cd
    >>> g = cd.DAG(arcs={(0, 1), (1, 3), (2, 3), (3, 4})
    >>> g.markov_blanket_of(1)
    {0, 2, 3}
    """
    ch_ = children(B, node)
    # print('chidren set is', ch_)
    if ch_:
        parents_of_children = set.union(*(parents(B,c) for c in ch_))
    else:
        parents_of_children = set()
    mb = parents(B, node) | children(B, node) | parents_of_children - {node}
    return (mb, len(mb))


# [docs]    def neighbors_of(self, nodes: NodeSet) -> Set[Node]:
#         """
#         Return all nodes that are adjacent to the node or set of nodes ``node``.
#
#         Parameters
#         ----------
#         nodes
#             A node or set of nodes.
#
#         See Also
#         --------
#         parents_of, children_of, markov_blanket_of
#
#         Examples
#         --------
#         >>> import causaldag as cd
#         >>> g = cd.DAG(arcs={(0,1), (0,2)})
#         >>> g.neighbors_of(0)
#         {1, 2}
#         >>> g.neighbors_of(2)
#         {0}
#         """
#         if isinstance(nodes, set):
#             return set.union(*(self._neighbors[n] for n in nodes))
#         else:
#             return self._neighbors[nodes].copy()
#
#

