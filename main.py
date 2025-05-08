# Import necessary libraries
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from astroquery.ipac.nexsci.nasa_exoplanet_archive import NasaExoplanetArchive

# --- Configuration ---
# You can adjust these parameters
MAX_MULTIPLANET_SYSTEMS_TO_PLOT = 3 # Max multi-planet systems (>=3 planets) to show
MAX_OTHER_SYSTEMS_TO_PLOT = 2       # Max other systems to show if no/few multi-planet found
DEFAULT_FIGURE_SIZE = (16, 12)      # Size of the plot

# --- Sample Data Definition ---
# Used as a fallback if the NASA Exoplanet Archive query fails or returns no data.
SAMPLE_DATA = {
    'hostname': ['Kepler-11', 'Kepler-11', 'Kepler-11', 'Kepler-11', 'Kepler-11', 'Kepler-11', # A known 6-planet system
                 'TRAPPIST-1', 'TRAPPIST-1', 'TRAPPIST-1', # A known 7-planet system (showing 3 for sample)
                 'HD 10180', 'HD 10180'], # Another multi-planet system
    'pl_name': ['Kepler-11 b', 'Kepler-11 c', 'Kepler-11 d', 'Kepler-11 e', 'Kepler-11 f', 'Kepler-11 g',
                'TRAPPIST-1 b', 'TRAPPIST-1 c', 'TRAPPIST-1 d',
                'HD 10180 c', 'HD 10180 d'],
    'sy_pnum': [6, 6, 6, 6, 6, 6, 7, 7, 7, 7, 7], # Number of planets in the system
    'discoverymethod': ['Transit', 'Transit', 'Transit', 'Transit', 'Transit', 'Transit',
                        'Transit', 'Transit', 'Transit',
                        'Radial Velocity', 'Radial Velocity'],
    'disc_year': [2011, 2011, 2011, 2011, 2011, 2011, 2016, 2016, 2016, 2010, 2010]
}
df_sample = pd.DataFrame(SAMPLE_DATA)

def fetch_exoplanet_data():
    """
    Fetches exoplanet data from the NASA Exoplanet Archive.
    Returns a Pandas DataFrame.
    """
    print("Attempting to query NASA Exoplanet Archive...")
    try:
        # Query for confirmed planets, focusing on default parameters.
        # We select hostname (star), planet name, number of planets in system,
        # discovery method, and discovery year.
        # We filter for systems with at least one star and one planet recorded (sy_snum > 0, sy_pnum > 0).
        query_adql = """
        SELECT hostname, pl_name, sy_pnum, discoverymethod, disc_year
        FROM pscomppars
        WHERE default_flag = 1 AND sy_snum > 0 AND sy_pnum > 0
        ORDER BY sy_pnum DESC, disc_year DESC, hostname
        """
        exoplanet_data_table = NasaExoplanetArchive.query_criteria(
            table="pscomppars",
            select="hostname, pl_name, sy_pnum, discoverymethod, disc_year",
            order="sy_pnum DESC, disc_year DESC, hostname" # Order to get more populated/recent systems first
        )
        
        if not exoplanet_data_table:
            print("No data returned from NASA Exoplanet Archive. Using sample data.")
            return df_sample.copy()
        
        df_all_systems = exoplanet_data_table.to_pandas()
        
        if df_all_systems.empty:
            print("Query successful but returned no data (empty table). Using sample data.")
            return df_sample.copy()
        
        print(f"Successfully retrieved {len(df_all_systems)} records from NASA Exoplanet Archive.")
        return df_all_systems

    except Exception as e:
        print(f"Error querying NASA Exoplanet Archive: {e}")
        print("Using sample data for demonstration.")
        return df_sample.copy()

def select_systems_for_visualization(df_all_systems):
    """
    Selects a subset of planetary systems for visualization from the full dataset.
    Prioritizes multi-planet systems.
    Returns a Pandas DataFrame.
    """
    df_final_selection = pd.DataFrame() # Initialize an empty DataFrame

    if df_all_systems.empty:
        print("Input data for selection is empty. Cannot select systems.")
        return df_final_selection

    # Prioritize systems with a higher number of planets (e.g., 3 or more)
    df_multi_planet = df_all_systems[df_all_systems['sy_pnum'] >= 3].copy()
    selected_hostnames_multi = df_multi_planet['hostname'].unique()

    hosts_to_plot = []
    if len(selected_hostnames_multi) > 0:
        hosts_to_plot.extend(selected_hostnames_multi[:min(len(selected_hostnames_multi), MAX_MULTIPLANET_SYSTEMS_TO_PLOT)])
        print(f"Selected {len(hosts_to_plot)} multi-planet system(s) (>=3 planets) for visualization.")
    
    # If we haven't selected enough systems, try to get other systems
    if len(hosts_to_plot) < (MAX_MULTIPLANET_SYSTEMS_TO_PLOT + MAX_OTHER_SYSTEMS_TO_PLOT):
        remaining_slots = (MAX_MULTIPLANET_SYSTEMS_TO_PLOT + MAX_OTHER_SYSTEMS_TO_PLOT) - len(hosts_to_plot)
        if remaining_slots > 0:
            # Get other unique hostnames not already selected
            other_hosts = df_all_systems[~df_all_systems['hostname'].isin(hosts_to_plot)]['hostname'].unique()
            if len(other_hosts) > 0:
                additional_hosts = other_hosts[:min(len(other_hosts), remaining_slots)]
                hosts_to_plot.extend(additional_hosts)
                print(f"Added {len(additional_hosts)} additional system(s) for visualization.")

    if hosts_to_plot:
        df_final_selection = df_all_systems[df_all_systems['hostname'].isin(hosts_to_plot)].copy()
        print(f"Total systems selected for plot: {len(hosts_to_plot)} ({', '.join(hosts_to_plot)})")
    else:
        print("Could not select any specific host systems based on criteria. Visualization might be empty or use minimal sample.")
        # Fallback to a single system from the sample data if everything else fails and df_all_systems was sample
        if df_all_systems.equals(df_sample) and not df_sample.empty:
             first_sample_host = df_sample['hostname'].unique()[0]
             df_final_selection = df_sample[df_sample['hostname'] == first_sample_host].copy()
             print(f"Using first system from sample data as a last resort: {first_sample_host}")


    return df_final_selection

def build_graph(df_for_graph):
    """
    Builds a NetworkX graph from the selected exoplanet data.
    Returns a NetworkX Graph object.
    """
    G = nx.Graph()
    if df_for_graph.empty:
        print("Data for graph building is empty. Returning an empty graph.")
        return G

    for _, row in df_for_graph.iterrows():
        # Ensure data types and handle potential missing values gracefully
        star_name = str(row.get('hostname', 'Unknown Star'))
        planet_name = str(row.get('pl_name', 'Unknown Planet'))
        num_planets_in_system = int(row.get('sy_pnum', 0))
        discovery_method = str(row.get('discoverymethod', 'N/A'))
        discovery_year = str(row.get('disc_year', 'N/A'))

        # Add star node (if it doesn't exist)
        if not G.has_node(star_name):
            G.add_node(star_name, type='star', num_planets=num_planets_in_system)
            
        # Add planet node (if it doesn't exist)
        if not G.has_node(planet_name):
            G.add_node(planet_name, type='planet', 
                       discoverymethod=discovery_method, 
                       disc_year=discovery_year,
                       host_star=star_name) # Store host star for context if needed
            
        # Add edge between star and planet
        G.add_edge(star_name, planet_name)
    
    print(f"Graph built with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges.")
    return G

def visualize_graph(G, df_plotted_systems, is_sample_data):
    """
    Visualizes the NetworkX graph using Matplotlib.
    """
    if G.number_of_nodes() == 0:
        print("Graph is empty, nothing to visualize.")
        return

    plt.figure(figsize=DEFAULT_FIGURE_SIZE)
    
    # Attempt Kamada-Kawai layout; fall back to spring layout if it fails
    try:
        pos = nx.kamada_kawai_layout(G)
    except nx.NetworkXError as layout_error: 
        print(f"Kamada-Kawai layout failed ({layout_error}), falling back to spring layout.")
        pos = nx.spring_layout(G, k=0.9, iterations=70, seed=42) # Seed for reproducibility

    # Define node properties
    node_colors_map = []
    node_sizes_map = []
    node_labels = {}

    for node, attr in G.nodes(data=True):
        if attr.get('type') == 'star':
            node_colors_map.append('deepskyblue')
            node_sizes_map.append(3500 + attr.get('num_planets', 0) * 200) # Size star by num_planets
            node_labels[node] = f"{node}\n({attr.get('num_planets', '?')} planets)"
        elif attr.get('type') == 'planet':
            node_colors_map.append('salmon')
            node_sizes_map.append(1200)
            node_labels[node] = f"{node}\n(Discovered: {attr.get('disc_year', 'N/A')})"
        else: # Fallback for nodes with no type (should not happen with current logic)
            node_colors_map.append('lightgray')
            node_sizes_map.append(500)
            node_labels[node] = node


    # Draw nodes, edges, and labels
    nx.draw_networkx_nodes(G, pos, node_color=node_colors_map, node_size=node_sizes_map, alpha=0.9)
    nx.draw_networkx_edges(G, pos, edgelist=G.edges(), width=1.2, alpha=0.5, edge_color='grey')
    nx.draw_networkx_labels(G, pos, labels=node_labels, font_size=9, font_weight='normal')

    # Create a title for the plot
    title_source = "Sample Data"
    if not is_sample_data and not df_plotted_systems.empty:
        plotted_host_names = df_plotted_systems['hostname'].unique()
        if len(plotted_host_names) > 0:
             title_source = f"Systems: {', '.join(plotted_host_names)}"
    
    plt.title(f'Visualization of Exoplanet Systems\n({title_source})', fontsize=18, pad=20)
    
    # Create custom legend
    star_patch = plt.Line2D([0], [0], linestyle="none", marker="o", markersize=10, markerfacecolor="deepskyblue", label="Star")
    planet_patch = plt.Line2D([0], [0], linestyle="none", marker="o", markersize=10, markerfacecolor="salmon", label="Planet")
    plt.legend(handles=[star_patch, planet_patch], loc='best', fontsize='medium', frameon=True, facecolor='white', framealpha=0.7)

    plt.axis('off') # Hide axis
    plt.tight_layout() # Adjust plot to ensure everything fits without overlapping
    # plt.show()
    plt.savefig("planets.png", dpi=300)

# --- Main Execution ---
if __name__ == "__main__":
    # 1. Fetch data (from NASA Archive or sample if failed)
    all_data = fetch_exoplanet_data()
    
    # Determine if the data used is the sample data
    # This check is a bit simplistic; if API returns data identical to sample, it would be true.
    # A more robust check would be a flag from fetch_exoplanet_data. For now, this is okay.
    using_sample = all_data.equals(df_sample)
    if using_sample and not fetch_exoplanet_data.__defaults__: # A bit of a hack to see if it fell back
         print("Note: Visualization is based on the fallback sample data.")
    elif not all_data.empty and not using_sample:
         print("Note: Visualization is based on data queried from NASA Exoplanet Archive.")


    # 2. Select a subset of systems for visualization
    systems_to_plot_df = select_systems_for_visualization(all_data)

    # 3. Build the graph
    # If selection results in empty, but original data (sample) was not, use a bit of sample.
    if systems_to_plot_df.empty and not df_sample.empty and all_data.equals(df_sample):
        print("System selection was empty; falling back to the first system in sample data for graph.")
        first_host = df_sample['hostname'].unique()[0]
        systems_to_plot_df = df_sample[df_sample['hostname'] == first_host].copy()
        using_sample = True # Explicitly set this as we are now definitely using sample data

    exoplanet_graph = build_graph(systems_to_plot_df)
    
    # 4. Visualize the graph
    visualize_graph(exoplanet_graph, systems_to_plot_df, using_sample)

    print("\n--- Script Finished ---")
    print("Notes for users:")
    print("- Ensure you have an internet connection for live NASA data.")
    print("- Required libraries: pandas, networkx, matplotlib, astroquery.")
    print("  Install using: pip install pandas networkx matplotlib astroquery")


