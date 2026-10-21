from pathlib import Path
import zipfile
import re

import networkx as nx
import numpy as np
from geopy.distance import distance as geodesic_distance
from pyproj import Transformer

from src.utils.utils import download_from_url
from src.scenario.config import BaseConfig


TOPOLOGYZOO_DATA_URL = 'http://www.topology-zoo.org/files/archive.zip'
TOPOLOGYZOO_DATA_PATH = Path('data/topologyzoo')
ARCHIVE_NAME = 'archive.zip'
LINK_LABEL_REGEX = r"(?:[0-9]+\s?(?:x|\*)\s?)?(?:[0-9]+|[0-9]?\.[0-9]+)\s*(?:Gbps|Gbit\/s|Gb\/s|Mbps|Mbit\/s|Mb\/s)"
GBPS_REGEX = r"Gbps|Gbit\/s|Gb\/s"
MBPS_REGEX = r"Mbps|Mbit\/s|Mb\/s"

# traffic scaling factors for the topology zoo graphs. Manually obtained by checking how busy the links become when
# running a PackeRL simulation the topology (you see that this can't be too accurate, but it may be a good first guess).
TZ_TRAFFIC_SCALING = {'Aarnet': 7.2316866243438165, 'Abilene': 20, 'Abvt': 40.044684722713455,
                      'Aconet': 14.492484819867911, 'Agis': 5.883620146158152, 'Ai3': 23.387596074202406,
                      'Amres': 32.93279661393244, 'Ans': 4.720097713242381, 'Arn': 5.648257690570683,
                      'Arpanet196912': 4.093736266238203, 'Arpanet19706': 3.9010600486765106,
                      'Arpanet19719': 8.883252857700535, 'Arpanet19723': 5.098695503213087,
                      'Arpanet19728': 4.4562925145907535, 'Atmnet': 13.048443778667604, 'Basnet': 4.051367931064576,
                      'Bbnplanet': 9.4930004477759, 'Bellsouth': 27.91087216147621,
                      'BeyondTheNetwork': 43.32840299183874, 'Bics': 9.434424135143788, 'Biznet': 22.567161691421877,
                      'Bren': 21.877103784845147, 'BtAsiaPac': 24.104903351914096, 'BtEurope': 42.14788341638781,
                      'BtNorthAmerica': 71.51724205429915, 'Canerie': 35.23812336396968, 'Carnet': 5.330457490134722,
                      'Cesnet1993': 3.7421243775888042, 'Cesnet1997': 8.480845640217813,
                      'Cesnet1999': 8.116340509001825, 'Cesnet2001': 6.849470880922327,
                      'Cesnet200304': 8.54809495199597, 'Cesnet200511': 13.30279116913139,
                      'Cesnet200603': 17.8269467154935, 'Cesnet200706': 9.815462934584414,
                      'Cesnet201006': 6.9920463383239655, 'Chinanet': 38.85504888884088, 'Claranet': 10.901710762798201,
                      'Columbus': 27.71664751855536, 'Compuserve': 15.281100789329074,
                      'CrlNetworkServices': 5.219848626691677, 'Cudi': 10.629919552690298, 'Cwix': 22.12720229736509,
                      'Cynet': 6.22516141405734, 'Darkstrand': 21.675417597161147, 'Dataxchange': 11.511513536704213,
                      'Dfn': 58.994467289106254, 'EliBackbone': 25.022634046152145, 'Epoch': 5.509430689917697,
                      'Ernet': 28.89165415463724, 'Evolink': 31.14563658737139, 'Fatman': 22.246390844580507,
                      'Forthnet': 22.33848123642311, 'Gambia': 5.8845318722963755, 'Garr199901': 10.202936419764429,
                      'Garr199904': 26.052927152757587, 'Garr199905': 17.941462492087837,
                      'Garr200109': 16.240154429117517, 'Garr200112': 24.733051896974, 'Garr200404': 17.135367981817875,
                      'Garr200908': 14.719141630798958, 'Garr200909': 22.456643814181483,
                      'Garr200912': 14.895640600831559, 'Garr201001': 19.92380435368123, 'Gblnet': 12.732133516739978,
                      'Geant2001': 30, 'Geant2009': 23.036190253968606, 'Geant2012': 42.67438216056504,
                      'Getnet': 6.116671986998308, 'Globalcenter': 51.049684874935465, 'Goodnet': 15.62779996413179,
                      'Grena': 10.31320742647135, 'Gridnet': 11.50509706169667, 'GtsCe': 31.294786088603807,
                      'GtsCzechRepublic': 27.67201965841663, 'GtsHungary': 28.032061428050635,
                      'GtsPoland': 25.40311886418076, 'GtsRomania': 25.05697312199106,
                      'GtsSlovakia': 20.653180968669975, 'HiberniaCanada': 12.284254522111821,
                      'HiberniaGlobal': 40.17857142857143, 'HiberniaIreland': 6.458092880519863,
                      'HiberniaUk': 14.117937637862125, 'HiberniaUs': 51.82341650671784,
                      'HostwayInternational': 6.772937401122318, 'HurricaneElectric': 6.646680695749974,
                      'Ibm': 7.883436157493948, 'Iinet': 43.1514686880039, 'Ilan': 4.498974754872667,
                      'Integra': 17.893342997480225, 'IowaStatewideFiberMap': 40.29086044510325,
                      'Iris': 16.45706306456152, 'Istar': 5.5552035448156225, 'Itnet': 8.18274065328795,
                      'Janetbackbone': 40.51224129693254, 'Jgn2Plus': 15.702439781378164,
                      'KentmanJul2005': 7.260224016666074, 'Kreonet': 8.158183693392484,
                      'LambdaNet': 27.894933644951767, 'Latnet': 31.344518490612902, 'Layer42': 10.493309737896656,
                      'Litnet': 5.96755216208932, 'Marnet': 16.132133371213687, 'Missouri': 34.55139218621253,
                      'Mren': 5.75762906714063, 'Napnet': 5.834566397372663, 'Navigata': 9.137279662974809,
                      'Netrail': 7.0572416559097855, 'NetworkUsa': 10.645869147185836, 'Niif': 11.712158042935329,
                      'Noel': 20.17083482412707, 'Nordu1989': 11.263466732740294, 'Nordu1997': 22.474215656900537,
                      'Nsfnet': 15.705239583905506, 'Oxford': 12.249296549261247, 'Packetexchange': 10.252754785478963,
                      'Peer1': 7.423176184948852, 'Pern': 25.812619502868063, 'PionierL1': 22.743571418113,
                      'Psinet': 13.893166471467241, 'Quest': 20.417777315971474, 'Renam': 4.534698392425832,
                      'Renater1999': 16.454460374861945, 'Renater2001': 7.163231681091018,
                      'Renater2004': 39.38135080353897, 'Renater2006': 23.412805492148458,
                      'Renater2008': 23.56093732017881, 'Renater2010': 27.749564236468558,
                      'Restena': 17.708867985192672, 'Reuna': 5.664726482970717, 'Rhnet': 14.411147050597112,
                      'Rnp': 15.339851264721732, 'RoedunetFibre': 29.15078057400576, 'Sago': 40.00000000000001,
                      'Sanet': 38.081805359661494, 'Sanren': 3.4056124200938376, 'Savvis': 10.174648410972706,
                      'Shentel': 30.624055531033523, 'Sinet': 7.4377282578510355, 'Singaren': 19.277386266748863,
                      'Spiralight': 5.906076548966916, 'Sprint': 7.37006076214309, 'Switch': 34.52685421994884,
                      'SwitchL3': 20.50772537542145, 'Syringa': 30.235162374020145, 'Telecomserbia': 6.485333127736711,
                      'Tinet': 33.096862387841426, 'Twaren': 15.700196505982166, 'Ulaknet': 11.244213243964632,
                      'Uninett2010': 2.9776192412119644, 'Uran': 7.236962947764791, 'UsCarrier': 31.034482758620683,
                      'Uunet': 50.290258449917935, 'Vinaren': 7.047114401363069, 'VisionNet': 14.212836825810406,
                      'VtlWavenet2008': 24.243811784064857, 'VtlWavenet2011': 27.110173022051093,
                      'WideJpn': 15.239539752073226, 'Xeex': 5.591873424700738, 'Xspedius': 16.406159749291827,
                      'York': 11.968838331709081}

# valid topology zoo graph topologies: no multi-graphs, no non-connected graphs (161 out of 261 Topologies)
VALID_TZ_TOPOLOGIES = ['Aarnet', 'Abilene', 'Abvt', 'Aconet', 'Agis', 'Ai3', 'Amres', 'Ans', 'Arn', 'Arpanet196912',
                       'Arpanet19706', 'Arpanet19719', 'Arpanet19723', 'Arpanet19728', 'Atmnet', 'Basnet', 'Bbnplanet',
                       'Bellsouth', 'BeyondTheNetwork', 'Bics', 'Biznet', 'Bren', 'BtAsiaPac', 'BtEurope',
                       'BtNorthAmerica', 'Canerie', 'Carnet', 'Cesnet1993', 'Cesnet1997', 'Cesnet1999', 'Cesnet2001',
                       'Cesnet200304', 'Cesnet200511', 'Cesnet200603', 'Cesnet200706', 'Cesnet201006', 'Chinanet',
                       'Claranet', 'Columbus', 'Compuserve', 'CrlNetworkServices', 'Cudi', 'Cwix', 'Cynet',
                       'Darkstrand', 'Dataxchange', 'Dfn', 'EliBackbone', 'Epoch', 'Ernet', 'Evolink', 'Fatman',
                       'Forthnet', 'Gambia', 'Garr199901', 'Garr199904', 'Garr199905', 'Garr200109', 'Garr200112',
                       'Garr200404', 'Garr200908', 'Garr200909', 'Garr200912', 'Garr201001', 'Gblnet', 'Geant2001',
                       'Geant2009', 'Geant2012', 'Getnet', 'Globalcenter', 'Goodnet', 'Grena', 'Gridnet', 'GtsCe',
                       'GtsCzechRepublic', 'GtsHungary', 'GtsPoland', 'GtsRomania', 'GtsSlovakia', 'HiberniaCanada',
                       'HiberniaGlobal', 'HiberniaIreland', 'HiberniaUk', 'HiberniaUs', 'HostwayInternational',
                       'HurricaneElectric', 'Ibm', 'Iinet', 'Ilan', 'Integra', 'IowaStatewideFiberMap', 'Iris',
                       'Istar', 'Itnet', 'Janetbackbone', 'Jgn2Plus', 'KentmanJul2005', 'Kreonet', 'LambdaNet',
                       'Latnet', 'Layer42', 'Litnet', 'Marnet', 'Missouri', 'Mren', 'Napnet', 'Navigata', 'Netrail',
                       'NetworkUsa', 'Niif', 'Noel', 'Nordu1989', 'Nordu1997', 'Nsfnet', 'Oxford', 'Packetexchange',
                       'Peer1', 'Pern', 'PionierL1', 'Psinet', 'Quest', 'Renam', 'Renater1999', 'Renater2001',
                       'Renater2004', 'Renater2006', 'Renater2008', 'Renater2010', 'Restena', 'Reuna', 'Rhnet', 'Rnp',
                       'RoedunetFibre', 'Sago', 'Sanet', 'Sanren', 'Savvis', 'Shentel', 'Sinet', 'Singaren',
                       'Spiralight', 'Sprint', 'Switch', 'SwitchL3', 'Syringa', 'Telecomserbia', 'Tinet', 'Twaren',
                       'Ulaknet', 'Uninett2010', 'Uran', 'UsCarrier', 'Uunet', 'Vinaren', 'VisionNet', 'VtlWavenet2008',
                       'VtlWavenet2011', 'WideJpn', 'Xeex', 'Xspedius', 'York']
NUM_VALID_TZ_TOPOLOGIES = len(VALID_TZ_TOPOLOGIES)

# projection to convert lat/lon to mercator coordinates
LAT_LON_TO_MERCATOR = Transformer.from_crs('EPSG:4326', 'EPSG:3857')

# special link names/types with known datarates
SPECIAL_LINK_DATARATES = {
    "STM-1": 155.52e6,
    "GigE": 1e9,
    "Fiber": 1e9,
    "Optic": 1e9,
    "T1": 1.544e6,
    "T3": 44.736e6,
    "E1": 2.048e6,
    "OC-192c": 9.95328e9,
    "9600 bit/s": 9600,
}


class TopologyZooConfig(BaseConfig):
    """
    Configuration for Topology Zoo graph generation.
    """
    in_order: bool
    network: str


def check_topologyzoo_data_available():
    """
    Checks whether the topology zoo data is available in the given data path. If not, downloads and extracts it.
    """
    tz_archive_path = TOPOLOGYZOO_DATA_PATH / ARCHIVE_NAME
    if not TOPOLOGYZOO_DATA_PATH.exists():
        TOPOLOGYZOO_DATA_PATH.mkdir(parents=True)
        try:
            download_from_url(TOPOLOGYZOO_DATA_URL, str(tz_archive_path.resolve()))
        except Exception as e:
            raise ValueError(f"Failed to download the topology zoo data: {e}")

    elif not TOPOLOGYZOO_DATA_PATH.is_dir():
        raise ValueError("a non-dir is sitting where the topologyzoo data dir is expected!")

    # unpack and delete the graph zoo archive if not yet done
    if tz_archive_path in TOPOLOGYZOO_DATA_PATH.glob('*'):
        with zipfile.ZipFile(tz_archive_path, 'r') as zip_file:
            zip_file.extractall(TOPOLOGYZOO_DATA_PATH)
        tz_archive_path.unlink()


def parse_link_speed_raw(link_speed_raw_attr):
    """
    Parses the raw link speed attribute from the topology zoo
    """
    return float(link_speed_raw_attr)


def parse_link_label(link_label_attr):
    """
    Parses the link label attribute from the topology zoo
    """
    matches = re.findall(LINK_LABEL_REGEX, link_label_attr)
    datarate = None
    if len(matches) > 0:
        # if there are multiple matches, take the one containing Gbps, Gbit/s, or Gb/s
        match = next(m for m in matches if re.search(GBPS_REGEX, m)) if len(matches) > 1 else matches[0]

        # split to get multiplier and raw datarate
        elems = re.split(r'x|\*', match)
        if len(elems) == 2:
            multiplier = int(''.join(c for c in elems[0] if c.isdigit() or c == '.'))
            raw_datarate = float(''.join(c for c in elems[1] if c.isdigit() or c == '.'))
        else:
            multiplier = 1
            raw_datarate = float(''.join(c for c in elems[0] if c.isdigit() or c == '.'))

        # get actual datarate
        datarate = multiplier * raw_datarate
        if re.search(GBPS_REGEX, match):
            datarate *= 1e9
        elif re.search(MBPS_REGEX, match):
            datarate *= 1e6

    return datarate


def obtain_link_datarate(link_attrs):
    """
    Obtains the link datarate from the given link attributes
    """
    if 'LinkSpeedRaw' in link_attrs:
        datarate = parse_link_speed_raw(link_attrs['LinkSpeedRaw'])
        if datarate is not None:
            return datarate

    if 'LinkLabel' in link_attrs:
        datarate = parse_link_label(link_attrs['LinkLabel'])
        if datarate is not None:
            return datarate
        else:  # check for special link names/types
            for special_link_name, special_link_datarate in SPECIAL_LINK_DATARATES.items():
                if special_link_name in link_attrs['LinkLabel']:
                    return special_link_datarate

    # no data found in LinkSpeedRaw or LinkLabel -> return mock value that gets replaced later on
    return -1


def load_tz_topology(tz_filename):
    """
    Loads a graph from the Topology Zoo dataset.
    """
    G_raw = nx.read_graphml(tz_filename, node_type=int)
    G = nx.from_scipy_sparse_array(nx.adjacency_matrix(G_raw))
    G.graph['name'] = G_raw.graph['label']

    # if node positions are available for all nodes, use them to obtain link delays
    # and convert them to 2D-embeddings using the mercator projection
    node_labels = set.intersection(*[set(attrs.keys()) for n, attrs in G_raw.nodes(data=True)])
    has_positions = 'Latitude' in node_labels and 'Longitude' in node_labels
    if has_positions:
        node_lat_lon = {n: (attrs['Latitude'], attrs['Longitude']) for n, attrs in G_raw.nodes(data=True)}

        # determine edge delays based on geodesic distances
        edge_distances = {(u, v): round(geodesic_distance(node_lat_lon[u], node_lat_lon[v]).km, 3)
                          for u, v in G_raw.edges()}
        max_distance = max(edge_distances.values())
        edge_delays = {e: 1 + round(9 * d / max_distance) for e, d in edge_distances.items()}
        nx.set_edge_attributes(G, edge_delays, 'delay')

        # convert node positions to 2D-embeddings using the mercator projection
        node_x_y = {n: LAT_LON_TO_MERCATOR.transform(lat, lon) for n, (lat, lon) in node_lat_lon.items()}
        x_min, x_max = min(x for x, _ in node_x_y.values()), max(x for x, _ in node_x_y.values())
        y_min, y_max = min(y for _, y in node_x_y.values()), max(y for _, y in node_x_y.values())
        x_range = x_max - x_min
        y_range = y_max - y_min
        normalized_node_x, normalized_node_y = {}, {}
        for n, (x, y) in node_x_y.items():
            normalized_node_x[n] = (x - x_min) / x_range * 2 - 1
            normalized_node_y[n] = (y - y_min) / y_range * 2 - 1
        pos = {n: np.array([normalized_node_x[n], normalized_node_y[n]]) for n in G.nodes}
        nx.set_node_attributes(G, pos, 'pos')

    # if we can get some link datarates, use them to set the edge datarates (fill remaining with average)
    link_datarates = {(u, v): obtain_link_datarate(attrs) for u, v, attrs in G_raw.edges(data=True)}
    valid_link_datarates = {e: d for e, d in link_datarates.items() if d > 0}
    if len(valid_link_datarates) > 0:
        avg_data_rate = sum(valid_link_datarates.values()) / len(valid_link_datarates)
        valid_link_datarates |= {e: avg_data_rate for e, d in link_datarates.items() if d <= 0}
        nx.set_edge_attributes(G, valid_link_datarates, 'datarate')

    return G


def generate_topology_zoo(in_order: bool, rng: np.random.Generator, rng_i: int, network: str = None, **kwargs):
    """
    Generates a random graph by choosing one from the graph zoo dataset and pre-processing it.
    """
    # preliminary checks
    check_topologyzoo_data_available()

    # choose a graph from the graph zoo dataset
    if network != "all":
        if network not in VALID_TZ_TOPOLOGIES:
            raise ValueError(f"Topology Zoo graph {network} is not a valid graph!")
        else:
            chosen_tz_graph = network
    elif in_order:
        chosen_tz_graph = VALID_TZ_TOPOLOGIES[rng_i % len(VALID_TZ_TOPOLOGIES)]
    else:
        chosen_tz_graph = rng.choice(VALID_TZ_TOPOLOGIES)
    tz_filename = str((TOPOLOGYZOO_DATA_PATH / f"{chosen_tz_graph}.graphml").resolve())

    # load the chosen graph and return it
    G = load_tz_topology(tz_filename)
    if not nx.is_connected(G) or isinstance(G, nx.MultiGraph):
        raise RuntimeError("Topology Zoo graph is not connected or is a multigraph "
                           "-> this should not happen because we have filtered those out previously!")
    return G
