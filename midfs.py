import pydot

sig_hijo = 2

def dfs(nodo, max_hijos, prof, actual_prof, grafo, target, ruta_actual=[]):
    global sig_hijo
    
    if nodo > 15 or actual_prof == prof:
        return

    for i in range(1, max_hijos + 1):
        valor_hijo = sig_hijo

        if valor_hijo < 15:
            nodo_hijo = pydot.Node(str(valor_hijo))
        elif valor_hijo == 15:
            nodo_hijo = pydot.Node(str(valor_hijo), shape="doublecircle")
        else:
            nodo_hijo = pydot.Node(str(valor_hijo), color="yellow")

        grafo.add_node(nodo_hijo)
        grafo.add_edge(pydot.Edge(str(nodo), str(valor_hijo)))
        

        if valor_hijo == target:
            ruta_actual += [nodo]
            #print(ruta_actual, valor_hijo)
            # Si encontramos el nodo objetivo, marcamos la ruta
            for nodo_ruta in ruta_actual + [valor_hijo]:
                grafo.get_node(str(nodo_ruta))[0].set('color', 'red')

        sig_hijo += 1
        
        dfs(valor_hijo, max_hijos, prof, actual_prof + 1, grafo, target, ruta_actual + [nodo])

def datos():
  max_hijos = int(input("Ingrese el maximo de hijos por nodo: "))
  max_prof = int(input("Ingrese la profundidad maxima del arbol: "))
  objetivo = int(input("Ingrese el objetivo: "))
  return max_hijos, max_prof, objetivo

inicio = 1
max_hijos, max_prof, target = datos()


g = pydot.Dot()
g.set_node_defaults(shape="circle")
nodo_inicio = pydot.Node(str(inicio))
g.add_node(nodo_inicio)

dfs(inicio, max_hijos, max_prof, 0, g, target)

g.write_png('grafo_con_ruta.png')
