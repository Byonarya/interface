import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
import networkx as nx
from scipy.optimize import minimize

st.set_page_config(layout="wide")


# Personnalisations du style de l'application
st.markdown(
    """
    <style>
    body, .stApp {
        background-color: #FFFFFF;
        color: #000000;
    }
    .stButton>button {
        color: #FFFFFF;
        background-color: #000000;
        display: flex;
        justify-content: center;
        align-items: center;
        margin: auto;
    }
    /* Changer la couleur des titres et headers de la sidebar */
    section[data-testid="stSidebar"] h1, 
    section[data-testid="stSidebar"] h2, 
    section[data-testid="stSidebar"] h3 {
        color: #e26a50;  /* Couleur du texte (orange vif ici) */
    }
    /* Changer la couleur des labels des widgets (radio, selectbox, etc.) */
    section[data-testid="stSidebar"] label {
        color: #e8a091;  /* Bleu */
        font-weight: bold;  /* Mettre en gras */
    }
    </style>
    """,
    unsafe_allow_html=True
)






















############# Page 1

def page_intro():
    

    

    col1, col2, col3 = st.columns([0.5, 3, 0.5])

    with col2:
        st.title("1. Introduction aux Réseaux de Neurones (Deep Learning)")
        st.write("Un réseau de neuronnes est un modèle mathématique et informatique créer en 1943 par Warren McCulloch et Walter Pitts. Il est inspiré du fonctionnement du cerveau humain, composé de plusieurs couches de neurones interconnectés, chacun de ces neurones est une unité de calcul qui prend en entrée un vecteur de données et renvoie une valeur de sortie. Les neurones sont organisés en couches, chaque couche étant composée de plusieurs neurones. ")
        st.subheader("Représentation d'un réseau de neuronne")

    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        

        graph_intro = nx.DiGraph() # Créer le graphe
        graph_intro.add_edges_from([(1, 2), (1, 3),
                                    (2, 4), (3, 4),
                                    (4, 5)]) # Ajouter les arêtes

        # Étiquettes des arêtes
        edge_labels_intro = {(1, 2): "x w1 + b1",
                            (1, 3): "x w2 + b2",
                            (2, 4): "x w3",
                            (3, 4): "x w4",
                            (4, 5): "+ b3"}
        graph_intro.add_node(6)

        # Positionner les nœuds
        pos_intro = {1: (0, 0.5),  # Input
                    2: (3, 0.55),  # Couche cachée 1
                    3: (3, 0.45),  # Couche cachée 2
                    4: (6, 0.5),  # Sum
                    5: (8, 0.5), # Output
                    6: (3, 0.6)}  # Subtitle

        # Nom des noeuds
        node_labels_intro = {1: "Input",
                            2: "ReLU",
                            3: "ReLU",
                            4: "Sum",
                            5: "Output",
                            6: "Couche cachée"}

        # Forme des noeuds
        node_shape_intro = {1: "s",
                            2: "o",
                            3: "o",
                            4: "d",
                            5: "s",
                            6: "h"}

        # Couleur des noeuds
        node_color_intro = {1: "indigo",
                            2: "green",
                            3: "green",
                            4: "lightgray",
                            5: "purple",
                            6: "white"}

        ## Dessiner le graphe
        fig, ax = plt.subplots(figsize=(9, 9))

        # Dessiner chaque groupe de nœuds séparément selon leur forme
        for shape in set(node_shape_intro.values()):
            nodes = [n for n in graph_intro.nodes if node_shape_intro[n] == shape]
            colors = [node_color_intro[n] for n in nodes] 
            nx.draw_networkx_nodes(graph_intro,
                                pos_intro,
                                nodelist = nodes,
                                node_shape = shape,
                                node_size = 3000,
                                node_color = colors,
                                ax = ax)

        # Dessiner les arêtes et labels
        nx.draw_networkx_edges(graph_intro,
                            pos_intro,
                            edge_color = "gray",
                            ax = ax)
        for node, label in node_labels_intro.items():
            color = "black" if label in ("Sum", "Couche cachée") else "white"  # Rouge pour "Sum", noir pour les autres
            nx.draw_networkx_labels(graph_intro,
                                    pos_intro,
                                    labels = {node: label},
                                    font_color = color,
                                    font_size = 10)
        nx.draw_networkx_edge_labels(graph_intro,
                                    pos_intro,
                                    edge_labels = edge_labels_intro,
                                    font_color = "black",
                                    font_size = 10,
                                    ax = ax)


        plt.axis("off")
        st.pyplot(fig)
    
    col1, col2, col3 = st.columns([0.5, 3, 0.5])

    with col2:
        st.markdown(
        """
        L'**<span style='color:white; background-color:indigo'>input</span>** est un vecteur qui passera dans chacun des chemins possibles du réseau de neurones, ici, il y en a deux. Sur chacune des branches à emprunter, des **<span style='color:black; background-color:gray'>poids (w)</span>** et des **<span style='color:black; background-color:gray'>biais (b)</span>** sont présents. Les poids sont les coefficients qui modulent l'importance de chaque entrée dans un neurone, en les multipliant, tandis que les biais ont un rôle plus minime, mais ils permettent avant tout d'activer les neurones si les entrées sont nulles. Chaque neurone a une **<span style='color:white; background-color:green'>fonction d'activation</span>** qui a le rôle essentiel d'introduire la non-linéarité dans le modèle. La fin d'un réseau de neurones se conclut par la moyenne de chacun des chemins empruntés pour pouvoir donner l'**<span style='color:white; background-color:purple'>output</span>** du modèle.
        """,
        unsafe_allow_html=True
    )
        st.markdown("La représentation du réseau de neurones ci-dessus peut être écrite mathématiquement comme ci-dessous :", unsafe_allow_html=True)

    st.latex(r'''
            \text{y =}
            \begin{pmatrix}
            max(0, x \cdot w1 + b1)  \cdot w3 \\
            + \\
            max(0, x \cdot w2 + b2)  \cdot w4 \\
            \end{pmatrix} \text{+ b3}''')
    
    col1, col2, col3 = st.columns([0.5, 3, 0.5])

    with col2:
        st.write("")
        st.write("")
        st.subheader("Différentes fonctions d'activation")
        st.markdown("Il existe de nombreuses fonctions d'activation, chacune ayant ses propres utilisations. Les plus connues et les plus généralisables sont les fonctions ReLU (Rectified Linear Unit), sigmoïde et softmax.", unsafe_allow_html=True)


    # Taille du vecteur
    size_vect = 100

    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    x = np.linspace(-10, 10, size_vect)
    y_var = sigmoid(x) + np.random.normal(scale=0.1, size = size_vect)

    def normal(x, mu, sigma):
        return 1 / (sigma * np.sqrt(2 * np.pi)) * np.exp(-0.5 * ((x - mu) / sigma) ** 2)

    y_normal_var = normal(x, 0, 2) + np.random.normal(scale=0.07, size = size_vect)


    y_random_var = np.random.uniform(low=-10, high=10, size=size_vect)


    def relu(x):
        return np.maximum(0, x)
    
    def softplus(x):
        return np.log(1 + np.exp(x))

    y_relu = relu(x)

    plt.figure(figsize=(15, 3))
    plt.subplot(1, 3, 1)
    plt.plot(x, y_relu, color='green', alpha=0.7, label="Fonction d'activation ReLU")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend(loc = "upper left")

    y_sigmoid = sigmoid(x)

    plt.subplot(1, 3, 2)
    plt.plot(x, y_sigmoid, color='green', alpha=0.7, label="Fonction d'activation Sigmoid")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend(loc = "upper left")

    def linear(x,a,b):
        return a*x+b

    y_softplus = softplus(x)

    plt.subplot(1, 3, 3)
    plt.plot(x, y_softplus, color='green', alpha=0.7, label="Fonction d'activation Softplus")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend(loc = "upper left")
    st.pyplot(plt)

    col1, col2, col3 = st.columns([1, 1, 1])
    with col1:
        st.latex("f(x) = max(0, x)")
    with col2:
        st.latex("f(x) = \dfrac{1}{1 + e^{-x}}")
    with col3:
        st.latex("f(x) = \log(1 + e^x)")

    col1, col2, col3 = st.columns([0.5, 3, 0.5])

    with col2:
        st.write("")
        st.markdown("Les fonctions d'activation portent ce nom, car elles permettent de savoir si un neurone est actif ou non. Par exemple, pour la fonction ReLU, le neurone est actif lorsque x est supérieur à 0. Pour certaines fonctions comme la Softplus, un neurone est toujours actif, mais avec un degré d'activation plus ou moins fort, car la fonction Softplus n'atteint jamais zéro.", unsafe_allow_html=True)


































# Page 2
def page_bouton_nn():


    col1, col2, col3 = st.columns([0.5, 3, 0.5])
    with col2:
        st.title("2. Fonctionnement d'un réseau de neuronnes")
        st.write("Dans cette partie, vous pouvez interagir avec les boutons **<span style='color:white; background-color:black'>Suivant</span>** et **<span style='color:white; background-color:black'>Précédent</span>** pour voir comment un réseau de neurone fonctionne à chaque étape. Ce réseau neuronal possède une couche avec 2 neurones, tous deux ayant une fonction d'activation Softplus. Il a pour but d'ajuster une courbe au jeu de donnée bleu. Mathématiquement, le modèle s'écrit comme suit :", unsafe_allow_html=True)
        st.latex(r'''
            \text{y =}
            \begin{pmatrix}
            log(1 + e^{(x \cdot (-34.4) + 2.14)}) \cdot (-1.30) \\
            + \\
            log(1 + e^{(x \cdot (-2.52) + 1.29)}) \cdot 2.28 \\
            \end{pmatrix} \text{+ (-0.58)}''')
        st.write("")
        st.markdown("Ce réseau neuronal comporte déjà des poids et des biais optimisés. Le point rouge sur le graphique montre la transformation de l'input en output le long du réseau neuronal. Deux points rouges apparaîtront lorsque le deuxième neurone sera activé. Les 2 dernières étapes montrent l'ajustement d'une courbe sur le jeu de données bleu.", unsafe_allow_html=True)
    # Fontion d'activation
    # ReLU
    def relu(z):
        return np.maximum(0, z)

    # Softplus
    def softplus(z):
        return np.log(1 + np.exp(z))

    # Sigmoïde
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    # Lineraire
    def linear(x):
        return x

    size_vect = 100
    x_activation = np.linspace(-10, 10, size_vect)


    # Créer le graphe
    graph_intro = nx.DiGraph()
    graph_intro.add_edges_from([(1, 2), (1, 3), (2, 4), (3, 4), (4, 5)])

    # Étiquettes des arêtes
    edge_labels_intro = {(1, 2): "x (-34.4) + 2.14",
                        (1, 3): "x (-2.52) + 1.29",
                        (2, 4): "x (-1.30)",
                        (3, 4): "x 2.28",
                        (4, 5): "+ (-0.58)"}

    # Ajouter un noeud supplémentaire
    graph_intro.add_node(6)

    # Positionner les nœuds
    pos_intro = {1: (0, 0.5),  # Input
                2: (3, 0.55),  # Couche cachée 1
                3: (3, 0.45),  # Couche cachée 2
                4: (6, 0.5),  # Sum
                5: (8, 0.5),  # Output
                6: (3, 0.6)}  # Subtitle

    # Nom des noeuds
    node_labels_intro = {1: "Input",
                        2: "Softplus",
                        3: "Softplus",
                        4: "Sum",
                        5: "Output",
                        6: "Couche cachée"}

    # Forme des noeuds
    node_shape_intro = {1: "s",
                        2: "o",
                        3: "o",
                        4: "d",
                        5: "s",
                        6: "h"}

    # Couleur des noeuds
    node_color_intro = {1: "indigo",
                        2: "green",
                        3: "green",
                        4: "lightgray",
                        5: "purple",
                        6: "white"}

    # Couleur des contours des noeuds
    node_edge_color_intro = {1: "black",
                            2: "black",
                            3: "black",
                            4: "black",
                            5: "black",
                            6: "white"}

    # Resultat neural network
    input_data = np.arange(0, 1, 0.001)
    data_step2 = (input_data * (-34.4)) + 2.14
    data_step3 = softplus(data_step2)
    data_step4 = data_step3 * (-1.30)
    data_step6 = (input_data * (-2.52)) + 1.29
    data_step7 = softplus(data_step6)
    data_step8 = data_step7 * 2.28
    data_step9 = (data_step4 + data_step8)/2
    data_step10 = data_step9 + (-0.58)
    output_data = data_step10

    # Initialiser l'étape dans l'état de la session
    if "step" not in st.session_state:
        st.session_state.step = 1

    def draw_graph(step):
        fig, ax = plt.subplots(figsize=(9, 9))
        
        # Dessiner chaque groupe de nœuds séparément selon leur forme
        for shape in set(node_shape_intro.values()):
            nodes = [n for n in graph_intro.nodes if node_shape_intro[n] == shape]
            colors = [node_color_intro[n] for n in nodes]
            edge_colors = [node_edge_color_intro[n] for n in nodes]
            nx.draw_networkx_nodes(graph_intro,
                                pos_intro,
                                nodelist = nodes,
                                node_shape = shape,
                                
                                node_size = 3000,
                                node_color = colors,
                                edgecolors = edge_colors,
                                ax = ax)

        # Dessiner les arêtes
        nx.draw_networkx_edges(graph_intro,
                            pos_intro,
                            edge_color = "gray",
                            ax = ax)
        
        # Appliquer les étapes
        # Par défaut, tous les éléments sont en couleur initiale
        node_edge_color_step = node_edge_color_intro.copy()
        edge_label_color_step = {key: "black" for key in edge_labels_intro}



        # Étapes
        if step == 1:
            node_edge_color_step[1] = "red"  # Contour du nœud 1 en rouge
        elif step == 2:
            edge_label_color_step[(1, 2)] = "red"  # Texte de l'arête (1,2) en rouge
        elif step == 3:
            node_edge_color_step[2] = "red"  # Contour du nœud 2 en rouge
        elif step == 4:
            edge_label_color_step[(2, 4)] = "red"  # Texte de l'arête (2,4) en rouge
        elif step == 5:
            node_edge_color_step[1] = "red"  # Contour du nœud 1 en rouge
        elif step == 6:
            edge_label_color_step[(1, 3)] = "red"  # Texte de l'arête (1,3) en rouge
        elif step == 7:
            node_edge_color_step[3] = "red"  # Contour du nœud 3 en rouge
        elif step == 8:
            edge_label_color_step[(3, 4)] = "red"  # Texte de l'arête (3,4) en rouge
        elif step == 9:
            node_edge_color_step[4] = "red"  # Contour du nœud 4 en rouge
            edge_label_color_step[(4, 5)] = "red"
        elif step == 10:
            node_edge_color_step[5] = "red"  # Contour du nœud 5 en rouge
        elif step == 11:
            node_edge_color_step[5] = "red"
        elif step == 12:
            node_edge_color_step[5] = "red"
        # Appliquer les couleurs modifiées
        for node, edge_color in node_edge_color_step.items():
            nx.draw_networkx_nodes(graph_intro,
                                pos_intro,
                                nodelist = [node],
                                node_shape = node_shape_intro[node],
                                node_size = 3000,
                                node_color = node_color_intro[node],
                                edgecolors = edge_color,
                                ax = ax)

        # Afficher les labels des nœuds
        for node, label in node_labels_intro.items():
            color = "black" if label in ("Sum", "Couche cachée") else "white"  # Si c'est "Sum" ou "Couche cachée", texte en rouge
            nx.draw_networkx_labels(graph_intro,
                                    pos_intro,
                                    labels = {node: label},
                                    font_color = color,
                                    font_size = 10)

        # Afficher les labels des arêtes avec la couleur modifiée
        for edge, label in edge_labels_intro.items():
            color = edge_label_color_step[edge]
            nx.draw_networkx_edge_labels(graph_intro,
                                        pos_intro,
                                        edge_labels = {edge: label},
                                        font_color = color,
                                        font_size = 10,
                                        ax = ax)

        plt.axis("off")
        st.pyplot(fig)

    # Jeu de données
    time_intro = [0, 0.02, 0.01,
                0.2, 0.4, 0.15,
                0.6, 0.7, 0.9]

    value_intro = [-0.2, -0.1, 0.2,
                    0.8, 0.35, 0.8,
                    0, 0, -0.2]

    data_red = [input_data[0], data_step2[0], data_step3[0], data_step4[0], [data_step4[0],input_data[0]], [data_step4[0],data_step6[0]],[data_step4[0],data_step7[0]],[data_step4[0],data_step8[0]],output_data[0], output_data[0],[output_data[0], output_data[30], output_data[90], output_data[180], output_data[360], output_data[550], output_data[650], output_data[750]],output_data]
    x_red = [0,0,0,0,[0,0],[0,0],[0,0],[0,0],0,0,[0,input_data[30], input_data[90], input_data[180], input_data[360], input_data[550], input_data[650], input_data[750]],input_data]

    def drawgraph_2(step):
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(time_intro, value_intro, marker="o", linestyle="", color="b")
        ax.set_xlabel("Temps")
        ax.set_ylabel("Valeur")
        ax.scatter(x_red[step-1], data_red[step-1], c = "r", marker = "o")
        # Affichage dans Streamlit
        ax.grid(True)
        st.pyplot(fig)


    # Création de deux colonnes pour afficher les graphes côte à côte
    col1, col2, col3, col4 = st.columns([1,2,2, 1])

    with col2:
        # Affichage du graphe de réseau
        draw_graph(st.session_state.step)

    with col3:
        st.write("")
        st.write("")
        st.write("")
        st.write("")
        st.write("")
        st.write("")
        st.write("")
        drawgraph_2(st.session_state.step)

    if "step" not in st.session_state:
        st.session_state.step = 1

    # Boutons de navigation
    col1, col2, col3 = st.columns([1,1,3])

    with col2:
        fig, ax = plt.subplots(figsize=(3, 2))
        ax.plot(softplus(x_activation), label="Softplus", color="green")
        ax.title.set_text("Fonction d'activation : Softplus")
        st.pyplot(fig, use_container_width=False)
        st.latex(r"f(x) = \log(1 + e^x)")

    with col3:
        # Espacement pour une meilleure lisibilité
        st.markdown(f"<h3 style='text-align: center;'>Étape actuelle : {st.session_state.step}</h3>", unsafe_allow_html=True)
        # Boutons pour naviguer entre les étapes
        if st.button("Suivant"):
            if st.session_state.step < 12:
                st.session_state.step += 1
        
        if st.button("Précédent"):
            if st.session_state.step > 1:
                st.session_state.step -= 1
    # Message d'introduction
    col1, col2, col3 = st.columns([0.5, 3, 0.5])
    with col2:
        st.write("")
        st.write("")
        st.write("")
        st.subheader("Que se passe-t-il avec un réseau neuronal plus complexe ?")
        st.markdown("Lorsque le réseau neuronal devient plus complexe, il prend en compte tous les chemins possibles de celui-ci. C'est donc là que la puissance du calcul numérique est nécessaire, afin de faire passer le vecteur d'entrée tout au long du réseau neuronal, mais aussi pour l'optimisation des poids et des biais.", unsafe_allow_html=True)

    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
    # Création du graphe du réseau de neurones
        G = nx.DiGraph()

        # Définition des couches
        layers = {
            "input": [0],  # 1 neurone d'entrée
            "hidden1": [1, 2, 3, 4],  # 4 neurones dans la couche cachée 1
            "hidden2": [5, 6, 7, 8],  # 4 neurones dans la couche cachée 2
            "hidden3": [9, 10, 11, 12],  # 4 neurones dans la couche cachée 3
            "output": [13],  # 1 neurone de sortie
        }

        # Nom des nœuds
        node_labels_complex = {
            0: "Input",
            13: "Output"
        }

        # Forme des nœuds (carré = entrée/sortie, cercle = couches cachées)
        node_shape_complex = {
            0: "s",  # carré
            13: "s",  # carré
        }

        # Couleur des nœuds
        node_color_complex = {
            0: "indigo",
            1: "green", 2: "green", 3: "green", 4: "green",
            5: "green", 6: "green", 7: "green", 8: "green",
            9: "green", 10: "green", 11: "green", 12: "green",
            13: "purple"
        }

        # Ajout des nœuds au graphe
        for layer_name, nodes in layers.items():
            for node in nodes:
                G.add_node(node, layer=layer_name)

        # Connexions entre les couches
        for (prev_layer, next_layer) in zip(list(layers.values())[:-1], list(layers.values())[1:]):
            for prev_node in prev_layer:
                for next_node in next_layer:
                    G.add_edge(prev_node, next_node)

        # Positionnement des neurones pour l'affichage
        pos = {}
        x_spacing = 2  # Espacement horizontal entre couches
        y_spacing = 1.5  # Espacement vertical entre neurones

        for i, (layer_name, nodes) in enumerate(layers.items()):
            y_start = -((len(nodes) - 1) * y_spacing) / 2  # Centrage vertical
            for j, node in enumerate(nodes):
                pos[node] = (i * x_spacing, y_start + j * y_spacing)

        # Dessin du graphe
        fig, ax = plt.subplots(figsize=(8, 6))

        # Dessiner chaque type de nœud séparément pour gérer les formes différentes
        for shape in set(node_shape_complex.values()):
            selected_nodes = [node for node, s in node_shape_complex.items() if s == shape]
            nx.draw_networkx_nodes(G, pos, nodelist=selected_nodes, node_shape=shape, 
                                node_color=[node_color_complex[n] for n in selected_nodes], 
                                node_size=1000, ax=ax)

        # Dessiner les nœuds standards (couches cachées)
        hidden_nodes = [node for node in G.nodes if node not in node_shape_complex]
        nx.draw_networkx_nodes(G, pos, nodelist=hidden_nodes, node_shape="o", 
                            node_color=[node_color_complex[n] for n in hidden_nodes], 
                            node_size=1000, ax=ax)

        # Dessiner les arêtes
        nx.draw_networkx_edges(G, pos, edge_color="gray", arrows=True, ax=ax)

        # Ajouter les labels des nœuds
        nx.draw_networkx_labels(G, pos, labels=node_labels_complex, font_size=10, font_color="white", ax=ax)

        # Affichage sur Streamlit
        ax.axis("off")
        st.pyplot(fig)

    col1, col2, col3 = st.columns([0.5, 3, 0.5])

    with col2:
        st.write("")
        st.markdown("Mathématiquement le réseau de neuronne ci-dessus peut s'écrire comme suit :", unsafe_allow_html=True)

        st.latex(r'''
        h_j = f\left( \sum_{k=1}^{4} x_k w_{kj} + b_j \right) \quad \text{(1ère couche cachée, 4 neurones)}
        ''')

        st.latex(r'''
        h_i = f\left( \sum_{j=1}^{4} h_j w_{ji} + b_i \right) \quad \text{(2ème couche cachée, 4 neurones)}
        ''')

        st.latex(r'''
        h_m = f\left( \sum_{i=1}^{4} h_i w_{im} + b_m \right) \quad \text{(3ème couche cachée, 4 neurones)}
        ''')

        st.latex(r'''
        y = \sum_{m=1}^{4} h_m w_{mo} + b_o \quad \text{(Sortie)}
        ''')

        st.latex(r'''
    x_k \text{ est l'entrée (avec k indice de 1 à 4)}, \, w_{kj} \text{ est le poids entre la couche d'entrée et la première couche cachée}, ''')
        st.latex(r'''\, b_j \text{ est le biais}, \, f \text{ est la fonction d'activation.}
        ''')
                 























def relu(z):
    return np.maximum(0, z)




# Optimisation via Descente de Gradient Stochastique avec Momentum
def sgd_optimizer(params, grads, lr=0.01, momentum=0.9, nesterov=False):
    if not hasattr(sgd_optimizer, "velocity"):
        sgd_optimizer.velocity = {'weights': [np.zeros_like(w) for w in params['weights']],
                                  'biases': [np.zeros_like(b) for b in params['biases']]}
    velocity = sgd_optimizer.velocity

    for i in range(len(params['weights'])):
        velocity['weights'][i] = momentum * velocity['weights'][i] - lr * grads['weights'][i]
        velocity['biases'][i] = momentum * velocity['biases'][i] - lr * grads['biases'][i]

        if nesterov:
            params['weights'][i] += momentum * velocity['weights'][i] - lr * grads['weights'][i]
            params['biases'][i] += momentum * velocity['biases'][i] - lr * grads['biases'][i]
        else:
            params['weights'][i] += velocity['weights'][i]
            params['biases'][i] += velocity['biases'][i]

    return params

def backpropagate(x, y, params, activation_function):
    m = len(x)
    grads = {'weights': [], 'biases': []}
    z = x.reshape(-1, 1)
    activations = [z]
    for W, b in zip(params['weights'], params['biases']):
        z = activation_function(np.dot(z, W) + b)
        activations.append(z)

    delta = activations[-1] - y.reshape(-1, 1)
    for i in range(len(params['weights'])-1, -1, -1):
        grads['weights'].insert(0, np.dot(activations[i].T, delta) / m)
        grads['biases'].insert(0, np.sum(delta, axis=0, keepdims=True) / m)
        delta = np.dot(delta, params['weights'][i].T) * (activations[i] > 0) 

    return grads
# 1. Fonction de récupération de la structure des paramètres
def get_params_structure(params):
    return {'weights': [w.shape for w in params['weights']], 'biases': [b.shape for b in params['biases']]}

# 2. Autres fonctions
def params_to_vector(params):
    vector = np.concatenate([w.flatten() for w in params['weights']] +
                            [b.flatten() for b in params['biases']])
    return vector

def vector_to_params(vector, params_structure):
    params = {'weights': [], 'biases': []}
    idx = 0
    for shape in params_structure['weights']:
        size = np.prod(shape)
        params['weights'].append(vector[idx:idx + size].reshape(shape))
        idx += size
    for shape in params_structure['biases']:
        size = np.prod(shape)
        params['biases'].append(vector[idx:idx + size].reshape(shape))
        idx += size
    return params

def forward(x, params, fonction):
    z = x.reshape(-1, 1)
    for W, b in zip(params['weights'], params['biases']):
        z = fonction(np.dot(z, W) + b)
    return z.flatten()

def cost(vector, x, y, params_structure, func=relu):
    # Convertir le vecteur en paramètres
    params = vector_to_params(vector, params_structure)
    # Faire une prédiction avec les paramètres actuels
    y_pred = forward(x, params, fonction=func)
    # Calculer la somme des carrés des erreurs
    return np.sum((y_pred - y) ** 2)

def cost_with_gradients(x, y, params, activation_function):
    # Convertir les paramètres en vecteur
    params_structure = get_params_structure(params)
    vector = params_to_vector(params)
    
    grads = backpropagate(x, y, params, activation_function)
    return cost(vector, x, y, params_structure), grads

def page_interactif():
    np.random.seed(0)

    def relu(z):
        return np.maximum(0, z)
    
    def linear(z):
        return z
    
    def tanh(x):
        return np.tanh(x)
    
    def softmax(x):
        e_x = np.exp(x - np.max(x))  # pour éviter le débordement numérique
        return e_x / e_x.sum(axis=0, keepdims=True)

    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    def normal(x, mu =0, sigma =2):
        return 1 / (sigma * np.sqrt(2 * np.pi)) * np.exp(-0.5 * ((x - mu) / sigma) ** 2)

    def softplus(z):
        return np.log(1 + np.exp(z))
    
    def leaky_relu(z, alpha=0.01):
        return np.where(z > 0, z, alpha * z)
    
    def swish(x):
        return x * sigmoid(x)
    
    def mish(x):
        return x * np.tanh(np.log(1 + np.exp(x)))
    
    def elu(z, alpha=1.0):
        return np.where(z > 0, z, alpha * (np.exp(z) - 1))
    
    def selu(z):
        return np.where(z > 0, 1.0507 * z, 1.67326 * (np.exp(z) - 1))

    def hard_sigmoid(x):
        return np.clip((x + 1) / 2, 0, 1)
    
    def hard_swish(x):
        return x * np.clip((x + 3) / 6, 0, 1)
    
    def gelu(x):
        return 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * np.power(x, 3))))





    def generate_data(size_vect=100, noise_scale=0.1, fonction=sigmoid):
        x = np.linspace(-10, 10, size_vect)
        y = fonction(x) + np.random.normal(scale=noise_scale, size=size_vect)
        x = (x - np.mean(x)) / np.std(x)
        return x, y

    def init_params(n_input, n_hidden, n_output, n_layers):
        np.random.seed(0)
        params = {'weights': [], 'biases': []}
        params['weights'].append(np.random.randn(n_input, n_hidden) * 0.1)
        params['biases'].append(np.zeros((1, n_hidden)))
        for _ in range(1, n_layers):
            params['weights'].append(np.random.randn(n_hidden, n_hidden) * 0.1)
            params['biases'].append(np.zeros((1, n_hidden)))
        params['weights'].append(np.random.randn(n_hidden, n_output) * 0.1)
        params['biases'].append(np.zeros((1, n_output)))
        return params

    def forward(x, params, fonction=relu):
        z = x.reshape(-1, 1)
        for W, b in zip(params['weights'], params['biases']):
            z = fonction(np.dot(z, W) + b)
        return z.flatten()

    def params_to_vector(params):
        vector = np.concatenate([w.flatten() for w in params['weights']] +
                                [b.flatten() for b in params['biases']])
        return vector

    def vector_to_params(vector, params_structure):
        params = {'weights': [], 'biases': []}
        idx = 0
        for shape in params_structure['weights']:
            size = np.prod(shape)
            params['weights'].append(vector[idx:idx + size].reshape(shape))
            idx += size
        for shape in params_structure['biases']:
            size = np.prod(shape)
            params['biases'].append(vector[idx:idx + size].reshape(shape))
            idx += size
        return params

    def get_params_structure(params):
        return {'weights': [w.shape for w in params['weights']], 'biases': [b.shape for b in params['biases']]}

    def draw_network(n_layers, n_hidden, ax):
        G = nx.DiGraph()
        
        # Ajouter les noeuds (input, hidden layers, output)
        for layer in range(n_layers + 2):
            num_nodes = n_hidden if 0 < layer < n_layers + 1 else 1
            for node in range(num_nodes):
                G.add_node((layer, node))

        # Ajouter les arêtes (connexions entre les noeuds)
        for layer in range(n_layers + 1):
            num_nodes_current = n_hidden if 0 < layer < n_layers else 1
            num_nodes_next = n_hidden if 0 < layer + 1 < n_layers + 1 else 1
            
            # Connexions entre la couche actuelle et la couche suivante
            for current_node in range(num_nodes_current):
                for next_node in range(num_nodes_next):
                    G.add_edge((layer, current_node), (layer + 1, next_node))

        # Assurer que la dernière couche cachée se connecte à l'output
        last_hidden_layer = n_layers  # La dernière couche cachée
        output_layer = n_layers + 1  # La couche de sortie
        num_hidden_nodes = n_hidden  # Nombre de noeuds dans la dernière couche cachée
        for node in range(num_hidden_nodes):
            G.add_edge((last_hidden_layer, node), (output_layer, 0))

        # Positionner les noeuds
        pos = {}
        for layer in range(n_layers + 2):
            num_nodes = n_hidden if 0 < layer < n_layers + 1 else 1
            for node in range(num_nodes):
                pos[(layer, node)] = (layer, -node + (num_nodes - 1) / 2.0)

        # Dessiner le graphe
        nx.draw(G, pos, with_labels=False, node_size=1000, node_color="lightblue", ax=ax)
        nx.draw_networkx_labels(G, pos, {(0, 0): "Input", (n_layers + 1, 0): "Output"}, ax=ax)
    
    
    
    
    st.title("3. Optimisation de courbe")
    st.sidebar.header("Configuration du modèle")

    n_layers = st.sidebar.slider("Nombre de couches cachées", 1, 5, 3)
    n_hidden = st.sidebar.slider("Nombre de nœuds par couche", 1, 10, 5)
    vect_size = st.sidebar.slider("Taille jeu de donnée (nombre de point)", 10, 100, 100)
    activation_function = st.sidebar.selectbox("Fonction d'activation", ["ReLU", "Mish", "ELU"])

    x, y_var = generate_data(size_vect=vect_size, noise_scale=0.05, fonction=sigmoid)
    n_input = 1
    n_output = 1
    params = init_params(n_input, n_hidden, n_output, n_layers)
    params_structure = get_params_structure(params)
    initial_vector = params_to_vector(params)
    cout = cost(initial_vector, x, y_var, params_structure, func=relu if activation_function == "ReLU" else sigmoid if activation_function == "Sigmoid" else softplus if activation_function == "Softplus" else linear if activation_function == "Linear" else tanh if activation_function =="Tanh" else softmax if activation_function == "Softmax" else leaky_relu if activation_function == "Leaky ReLU" else swish if activation_function == "Swish" else gelu if activation_function == "GELU" else mish if activation_function == "Mish" else elu if activation_function == "ELU" else selu if activation_function == "SELU" else hard_sigmoid if activation_function == "Hard Sigmoid" else hard_swish)


    # Partie d'optimisation
    num_iterations = 10000
    learning_rate = 0.1
    for i in range(num_iterations):
        cost_value, grads = cost_with_gradients(x, y_var, params, activation_function=relu)
        params = sgd_optimizer(params, grads, lr=learning_rate, momentum=0.9, nesterov=True)
        if i % 100 == 0:
            print(f"Iteration {i}/{num_iterations}, Cost: {cost_value}")

    # Prédiction du modèle optimisé
    y_fit = forward(x, params, fonction=relu if activation_function == "ReLU" else sigmoid if activation_function == "Sigmoid" else softplus if activation_function == "Softplus" else linear if activation_function == "Linear" else tanh if activation_function=="Tanh" else softmax if activation_function == "Softmax" else leaky_relu if activation_function == "Leaky ReLU" else swish if activation_function == "Swish" else gelu if activation_function == "GELU" else mish if activation_function == "Mish" else elu if activation_function == "ELU" else selu if activation_function == "SELU" else hard_sigmoid if activation_function == "Hard Sigmoid" else hard_swish)

    col1, col2 = st.columns([1, 1.25])

    with col1:
        st.subheader("Structure du réseau de neurones")
        fig, ax = plt.subplots(figsize=(8, 6))
        draw_network(n_layers, n_hidden, ax)
        st.pyplot(fig)
    with col2:
        st.subheader("Résultats de l'ajustement de la courbe")
        fig, ax = plt.subplots()
        ax.scatter(x, y_var, label="Données", alpha=0.7)
        ax.plot(x, y_fit, color='red', label="Ajustement")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.legend()
        ax.grid()
        st.pyplot(fig)
    
    y_relu = relu(x)

    y_ELU = elu(x)

    y_Mish = mish(x)

        # Partie du code de la visualisation des fonctions d'activation
    activation_highlight = activation_function  # Récupérer la fonction d'activation choisie

    plt.figure(figsize=(15, 3))

    # Première fonction : ReLU
    plt.subplot(1, 3, 1)
    plt.plot(x, y_relu, color='green', alpha=0.7, label="Fonction d'activation ReLU")
    if activation_highlight == "ReLU":
        plt.gca().spines['top'].set_color('red')
        plt.gca().spines['top'].set_linewidth(2)
        plt.gca().spines['right'].set_color('red')
        plt.gca().spines['right'].set_linewidth(2)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend(loc="upper left")

    # Deuxième fonction : ELU
    plt.subplot(1, 3, 2)
    plt.plot(x, y_ELU, color='green', alpha=0.7, label="Fonction d'activation ELU")
    if activation_highlight == "ELU":
        plt.gca().spines['top'].set_color('red')
        plt.gca().spines['top'].set_linewidth(2)
        plt.gca().spines['right'].set_color('red')
        plt.gca().spines['right'].set_linewidth(2)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend(loc="upper left")

    # Troisième fonction : Mish
    plt.subplot(1, 3, 3)
    plt.plot(x, y_Mish, color='green', alpha=0.7, label="Fonction d'activation Mish")
    if activation_highlight == "Mish":
        plt.gca().spines['top'].set_color('red')
        plt.gca().spines['top'].set_linewidth(2)
        plt.gca().spines['right'].set_color('red')
        plt.gca().spines['right'].set_linewidth(2)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend(loc="upper left")

    st.pyplot(plt)

    col1, col2, col3 = st.columns([0.5, 3, 0.5])

    with col2:
        st.markdown("La fonction d'activation ReLU (Rectified Linear Unit) est très couramment utilisée, car elle est facile à implémenter (présente dans la majorité des packages de machine learning), rapide en termes de calcul et assez généraliste pour de nombreux problèmes.", unsafe_allow_html=True)
        st.markdown("La fonction d'activation ELU (Exponential Linear Unit) est une modification de ReLU qui est très sensible à l'initialisation des poids avant l'apprentissage. Elle est majoritairement utilisée pour des réseaux de neurones convolutifs (CNN) et pour des tâches de régression.", unsafe_allow_html=True)
        st.markdown("La fonction d'activation Mish est une combinaison de ReLU et de la sigmoïde, donc bien plus coûteuse en temps de calcul.", unsafe_allow_html=True)




















def page_end(): 
        col1, col2, col3 = st.columns([0.5, 3, 0.5])  
        with col2:

            st.title("4. Conclusion")
            st.write("Cette application a pour but de vous montrer comment fonctionne un réseau de neurones, à la fois visuellement et mathématiquement.")  
            st.write("L'application des réseaux de neurones pour une tâche de régression est peu courante en raison des performances des modèles de régression classiques. Cependant, cette approche permet de mieux comprendre leur fonctionnement. La majorité des utilisations du deep learning concernent l'analyse de texte ou d'images, avec des modèles comme les transformers ou les CNN, qui permettent de traiter une ou plusieurs matrices en entrée pour l'analyse d'images.")  
            st.write("L'image ci-dessous illustre l'utilisation d'un CNN pour détecter des zones d'eau, de corail et de sable sur une image satellite.")  
            st.image("https://www.cell.com/cms/10.1016/j.tree.2019.03.006/asset/b0d6cd3f-7902-4162-8998-513fd646463d/main.assets/gr3_lrg.jpg", caption= "Brodrick et al., 2019")
            



























def page_remerciement():
    col1, col2, col3 = st.columns([0.5, 3, 0.5])
    with col2:
        st.title("5. Références et Journal de bord")
        st.subheader("Références")
        st.write("Brodrick PG, Davies AB, Asner GP. Uncovering Ecological Patterns with Convolutional Neural Networks. Trends Ecol Evol. 2019 Aug;34(8):734-745. doi: 10.1016/j.tree.2019.03.006. Epub 2019 May 8. PMID: 31078331.")
        st.write("https://www.youtube.com/watch?v=CqOfi41LfDw")
        st.subheader("Journal de bord")
        st.write("Les améliorations à apporter à l'application seraient, tout d'abord, l'optimisation de la structure de conditions (`if`) utilisées pour la page 2, qui n'est pas optimale et peut entraîner de légers bugs. J'aurais également aimé inclure plus d'informations et d'animations pour expliquer la backpropagation ainsi que le fonctionnement des CNN et/ou des transformers. Je n'ai pas compté le nombre d'heures passées sur ce projet, mais l'activité la plus chronophage a été la mise en place du réseau de neurones dynamique sur la page 3.")  
        st.write("")
        st.write("")
        st.write("")
        st.write("Marion Sésia, février 2025")
































st.sidebar.title("Navigation")
choix = st.sidebar.radio("", ["1. Introduction", "2. Fonctionnement d'un réseau de neuronnes", "3. Optimisation de courbe", "4. Conclusion", "5. Références et Journal de bord"])
if choix == "1. Introduction":
    page_intro()
elif choix == "2. Fonctionnement d'un réseau de neuronnes":
    page_bouton_nn()
elif choix == "3. Optimisation de courbe":
    page_interactif()
elif choix == "4. Conclusion":
    page_end()
elif choix == "5. Références et Journal de bord":
    page_remerciement()
