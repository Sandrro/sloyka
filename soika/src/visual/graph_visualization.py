import folium
import random


def visualize_graph(G, zoom: int):
    """
    Visualizes a graph from the given graph_path using Folium and MarkerCluster.
    
    Args:
        output_file (str, optional): The file to save the visualization to. Defaults to None.
        
    Returns:
        folium.Map: The folium map object representing the visualized graph.
    """

    m = folium.Map(
        zoom_start=zoom,
        tiles="cartodb_positron",
        control_scale=True,
    )

    nodes_group = folium.FeatureGroup(name='Nodes')
    neighbors_group = folium.FeatureGroup(name='Neighbors')
    lines_group = folium.FeatureGroup(name='Lines')

    for node, data in G.nodes(data=True):
        if "Lat" in data and "Lon" in data:
            main_node_location = [data["Lon"], data["Lat"]]
            folium.Marker(
                location=main_node_location,
                popup=node,
                icon=folium.DivIcon(
                    icon_size=(60,20),
                    icon_anchor=(0,0),
                    html='<div style="font-size: 6pt; color: blue">%s</div>' % node,
                )
            ).add_to(nodes_group)
            
            for n in G.neighbors(node):
                neighbor_location = [
                    main_node_location[0] + random.uniform(-0.0008, 0.0008),
                    main_node_location[1] + random.uniform(-0.0008, 0.0008),
                ]
                folium.Marker(
                    location=neighbor_location,
                    popup=n,
                    icon=folium.DivIcon(
                        icon_size=(150,50),
                        icon_anchor=(0,0),
                        html=f'<div style="font-size: 10pt;color: darkred">{n}</div>',
                    )
                ).add_to(neighbors_group)
                
                folium.PolyLine([main_node_location, neighbor_location], color="purple", weight=1, opacity=1).add_to(lines_group)

   
    m.add_child(nodes_group)
    m.add_child(neighbors_group)
    m.add_child(lines_group)

    folium.LayerControl().add_to(m)
    
    return m
