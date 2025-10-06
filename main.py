import pygame
import osmnx as ox
import networkx as nx
import numpy as np
import os
from PIL import Image
from shapely.geometry import Polygon, LineString, MultiLineString
from geopy.distance import distance as geopy_distance, geodesic
from visualization import trace_greedy_bfs, trace_dijkstra, trace_a_star, trace_bellman_ford, trace_bfs, trace_dfs, draw_complexity_info
import math
from complexity import algorithm_complexities

# ----------------------------------------------------------------------------- 
# Initialize Pygame
pygame.init()
# Hand-cursor for hovering usecases
HAND_CURSOR = pygame.SYSTEM_CURSOR_HAND
ARROW_CURSOR = pygame.SYSTEM_CURSOR_ARROW

# Setting game caption 
pygame.display.set_caption("Pathfinding Algorithims In Bangalore City Map")

# Initializing game-window resolution (fixed)
info = pygame.display.Info()
screen_width, screen_height = info.current_w, info.current_h
screen = pygame.display.set_mode((screen_width, screen_height))
screen.fill((0, 0, 0, 0))
# ----------------------------------------------------------------------------- 


# Defining colors
ALT_HEADING_TEXT_COLOR = (87, 9, 135)  # Shade of purple
HEADING_TEXT_COLOR = (67, 56, 120)  # Shade of purple
ALT_BUTTON_BG_COLOR = (87, 9, 135, 128)  # Shade of purple made transparent
BUTTON_BG_COLOR = (67, 56, 120, 128 + 64)  # Shade of purple made transparent
ROAD_COLOR = (50, 50, 50)  # Dark gray roads
NODE_COLOR = (244, 208, 240)
START_COLOR = (255, 0, 0)  # Red start point
END_COLOR = (0, 255, 0)  # Green end point
TRACE_COLOR = (64, 224, 208)  # Turquoise
FINAL_PATH_COLOR = (255, 0, 0)  # Red final path
LABEL_COLOR = (50, 50, 50)  # Dark gray labels
BUILDING_COLOR = (235, 235, 235)  # Light gray background
BACKGROUND_COLOR = (255, 255, 255)
POND_COLOR = (173, 216, 230)  # Light blue for ponds
PARK_COLOR = (144, 238, 144)  # Light green for parks
WOOD_COLOR = (144, 238, 144)  # Light forest green for wood
WETLAND_COLOR = (175, 238, 238)  # Light aquamarine for wetland
GRASS_COLOR = (173, 255, 47)  # Pastel green for grass
FOREST_COLOR = (85, 107, 47)  # Olive drab for forest
TREE_COLOR = (144, 238, 144)  # Light green for trees
BEACH_COLOR = (255, 228, 196)  # Bisque for beaches
CLIFF_COLOR = (210, 180, 140)  # Tan for cliffs
FARMLAND_COLOR = (255, 248, 220)  # Cornsilk for farmland
ORCHARD_COLOR = (255, 228, 181)  # Moccasin for orchards


# Loading fonts (fall back to default if custom fonts missing)
def safe_font(path, size):
    try:
        return pygame.font.Font(path, size)
    except Exception:
        return pygame.font.SysFont(None, size)

calgary_font = safe_font(os.path.join("fonts", "Calgary_DEMO.ttf"), 28)
vingue_rg_font = safe_font(os.path.join("fonts", "vinque rg.otf"), 64)


# Loading Images (wrap loads in try/except so missing assets don't crash)
def safe_image_load(path, convert_alpha=True):
    try:
        img = pygame.image.load(path)
        return img.convert_alpha() if convert_alpha else img.convert()
    except Exception:
        # create a placeholder surface
        s = pygame.Surface((32, 32), pygame.SRCALPHA)
        pygame.draw.rect(s, (200, 200, 200), s.get_rect(), border_radius=4)
        return s

menu_screen_bg_image = pygame.transform.scale(safe_image_load(os.path.join("backgrounds", "bangalore_map3.jpg")), screen.get_size())
ribbon_label_image = pygame.transform.scale(safe_image_load(os.path.join("images", "ribbon_chip.png")), (int(screen_width/1.3), 40))
ribbon_label_image_short = pygame.transform.scale(safe_image_load(os.path.join("images", "ribbon_chip.png")), (int(screen_width/3), 40))
start_marker_image = safe_image_load(os.path.join("images", "red_marker_32.png"))
end_marker_image = safe_image_load(os.path.join("images", "green_marker_32.png"))
fixed_marker_image = safe_image_load(os.path.join("images", "pin_32.png"))
bmsit_logo_image = safe_image_load(os.path.join("images", "bmsit_64.png"))
home_image = safe_image_load(os.path.join("images", "home_32.png"))
rail_factory_image = safe_image_load(os.path.join("images", "rail_factory_32.png"))
car_image = safe_image_load(os.path.join("images", "car_top.png"))
scooty_image = safe_image_load(os.path.join("images", "scooty_top.png"))
directions = [i * 22.5 for i in range(16)]  # 0..337.5 step 22.5 degrees (16 directions)
car_images = {angle: pygame.transform.rotate(car_image, -angle + 90) for angle in directions}
scooty_images = {angle: pygame.transform.rotate(scooty_image, -angle + 270) for angle in directions}
vehicle_images = car_images
vehicle_image = car_image


# Define colors for each algorithm's path
COLORS = {
    'dijkstra': (255, 69, 0),  # Bright Orange Red
    'greedy_bfs': (205, 130, 50),
    'a_star': (50, 205, 50),  # Lime Green
    'bellman_ford': (0, 191, 255),  # Deep Sky Blue
    'bfs': (255, 215, 0),  # Gold
    'dfs': (138, 43, 226)  # Blue Violet
}


# Define pastel colors and appropriate widths for each road type
road_types = {
    'motorway': {'color': (111, 111, 111), 'width': 18},
    'trunk': {'color': (111, 111, 111), 'width': 14},
    'primary': {'color': (131, 131, 131), 'width': 10},
    'secondary': {'color': (151, 151, 151), 'width': 7},
    'tertiary': {'color': (181, 181, 181), 'width': 4},
    'service': {'color': (211, 211, 211), 'width': 2},
    'residential': {'color': (211, 211, 211), 'width': 2},
    'unclassified': {'color': (211, 211, 211), 'width': 1},
    'pedestrian': {'color': (211, 211, 211), 'width': 1},
}


# Function to render text at a larger size and scale down - NOT SURE IF IT IMPROVED QUALITY
def render_text_scaled(font_name_or_obj, text, size, color):
    """Accepts either a font name string or a pygame.font.Font object."""
    if isinstance(font_name_or_obj, pygame.font.Font):
        base_font = font_name_or_obj
    else:
        base_font = pygame.font.SysFont(font_name_or_obj, max(10, int(size * 3)))
    large_font_size = max(10, size * 3)
    try:
        # create a large font surface then downscale
        large_font = base_font
        text_surface = large_font.render(text, True, color)
        scaled_surface = pygame.transform.smoothscale(text_surface, (max(1, text_surface.get_width() // 3), max(1, text_surface.get_height() // 3)))
        return scaled_surface
    except Exception:
        # fallback to a simple render
        fallback_font = pygame.font.SysFont(None, size)
        return fallback_font.render(text, True, color)


# Define the bounding box (left, bottom, right, top)
bbox = (77.550, 13.090, 77.650, 13.150)  # Coordinates in (left, bottom, right, top) order

# Create the graph using the correct function signature
G = ox.graph.graph_from_bbox(bbox[3], bbox[1], bbox[2], bbox[0], network_type='drive') if False else ox.graph.graph_from_bbox(bbox, network_type='drive')
# NOTE: osmnx has multiple function signatures across versions. The simpler call used below should usually work:
# G = ox.graph_from_bbox(north, south, east, west, network_type='drive') - but since you used graph_from_bbox(bbox, ...), keep that.
# The above toggled expression preserves your intended call; if your osmnx version requires a different signature, adapt accordingly.

# Convert graph to geodataframes
nodes, edges = ox.graph_to_gdfs(G) 
streets = ox.graph_to_gdfs(G, nodes=False) 


# Coordinates for BMSIT&M (Longitude, Latitude)
bmsitm_coords = (77.5689133820817, 13.13431180979886)
home_coords = (77.62529533502402, 13.143609724261097)


# Variables for vehicle path navigation
vehicle_path = []
current_vehicle_pos_index = 0
vehicle_path_updated = False


# Specifying tags for additional features such as natural elements (e.g., ponds) 
tags = {
    'natural': ['water', 'wood', 'wetland', 'tree', 'beach', 'cliff'],
    'landuse': ['residential', 'commercial', 'industrial', 'forest', 'meadow', 'grass', 'farmland', 'orchard', 'vineyard', 'park']
}
try:
    geometries = ox.features_from_bbox(bbox, tags=tags)
except Exception:
    # If features_from_bbox not available or fails, create empty GeoDataFrame-like
    import pandas as pd
    geometries = pd.DataFrame({'geometry': []})

minx, miny, maxx, maxy = nodes['x'].min(), nodes['y'].min(), nodes['x'].max(), nodes['y'].max()


def normalize(val, min_val, max_val, new_min, new_max):
    if max_val - min_val == 0:
        return (new_min + new_max) / 2
    return (val - min_val) / (max_val - min_val) * (new_max - new_min) + new_min


pos = {node: (normalize(data['x'], minx, maxx, 0, screen_width), normalize(data['y'], miny, maxy, 0, screen_height)) for node, data in G.nodes(data=True)}
# Initial start and end points
start = list(G.nodes())[0]
end = list(G.nodes())[-1]


# Pre-render the entire map and labels on a static surface
map_surface = pygame.Surface((screen_width, screen_height))
map_surface.fill(BACKGROUND_COLOR)


# Draw additional features (e.g., ponds, parks, wood, wetland, grass, forest, trees, beaches, cliffs, buildings, amenities, historic sites)
# Font for labeling
label_list = []
for _, geom in geometries.iterrows() if hasattr(geometries, "iterrows") else []:
    try:
        if isinstance(geom.geometry, Polygon):
            feature_coords = [(normalize(x, minx, maxx, 0, screen_width), normalize(y, miny, maxy, 0, screen_height)) for x, y in geom.geometry.exterior.coords]
            centroid_x = sum([coord[0] for coord in feature_coords]) / len(feature_coords)
            centroid_y = sum([coord[1] for coord in feature_coords]) / len(feature_coords)

            natural_tag = geom.get('natural', '')
            landuse_tag = geom.get('landuse', '')
            feature_name = geom.get('name', '')

            label_pos = (int(centroid_x), int(centroid_y))
            label = ''
            if isinstance(natural_tag, str):
                if 'water' in natural_tag:
                    pygame.draw.polygon(map_surface, POND_COLOR, feature_coords)
                elif 'wood' in natural_tag:
                    pygame.draw.polygon(map_surface, WOOD_COLOR, feature_coords)
                elif 'wetland' in natural_tag:
                    pygame.draw.polygon(map_surface, WETLAND_COLOR, feature_coords)
                elif 'tree' in natural_tag:
                    pygame.draw.polygon(map_surface, TREE_COLOR, feature_coords)
                elif 'beach' in natural_tag:
                    pygame.draw.polygon(map_surface, BEACH_COLOR, feature_coords)
                elif 'cliff' in natural_tag:
                    pygame.draw.polygon(map_surface, CLIFF_COLOR, feature_coords)

                if isinstance(feature_name, str) and natural_tag in ['wood', 'water', 'wetland', 'tree', 'beach', 'cliff']:
                    label = str(feature_name)

            elif isinstance(landuse_tag, str):
                if 'forest' in landuse_tag:
                    pygame.draw.polygon(map_surface, FOREST_COLOR, feature_coords)
                elif 'grass' in landuse_tag:
                    pygame.draw.polygon(map_surface, GRASS_COLOR, feature_coords)
                elif 'park' in landuse_tag:
                    pygame.draw.polygon(map_surface, PARK_COLOR, feature_coords)
                elif landuse_tag in ['commercial', 'residential']:
                    pygame.draw.polygon(map_surface, BUILDING_COLOR, feature_coords)
                elif 'farmland' in landuse_tag:
                    pygame.draw.polygon(map_surface, FARMLAND_COLOR, feature_coords)
                elif 'orchard' in landuse_tag:
                    pygame.draw.polygon(map_surface, ORCHARD_COLOR, feature_coords)

                if landuse_tag not in ['commercial', 'residential', 'grass']:
                    if isinstance(feature_name, str):
                        if 'yelahanka rail wheel factory' in feature_name.lower():
                            map_surface.blit(rail_factory_image, (centroid_x, centroid_y))
                            label = str(feature_name)

            if label:
                font_name = pygame.font.get_default_font()
                font_size = 16
                font_color = HEADING_TEXT_COLOR

                if label not in [items[0] for items in label_list]:
                    label_list.append([label, (centroid_x, centroid_y)])
    except Exception:
        continue


# Drawing roads as thicker lines with different colors based on road type
for _, row in edges.iterrows():
    try:
        road_type = 'unclassified'  # Default road type
        highway = row.get('highway', '')
        if isinstance(highway, list):
            for hwy in highway:
                if hwy in road_types:
                    road_type = hwy
                    break
        elif highway in road_types:
            road_type = highway
        road_color = road_types[road_type]['color']
        road_width = road_types[road_type]['width']
        if isinstance(row.geometry, (LineString, MultiLineString)):
            for line in row.geometry if isinstance(row.geometry, MultiLineString) else [row.geometry]:
                road_coords = [(normalize(x, minx, maxx, 0, screen_width), normalize(y, miny, maxy, 0, screen_height)) for x, y in line.coords]
                if len(road_coords) > 1:
                    # convert to integer tuples
                    int_coords = [(int(x), int(y)) for x, y in road_coords]
                    pygame.draw.lines(map_surface, road_color, False, int_coords, max(1, int(road_width)))
    except Exception:
        continue


# Drawing nodes as circles
for node in G.nodes():
    try:
        p = (int(pos[node][0]), int(pos[node][1]))
        pygame.draw.circle(map_surface, road_types['residential']['color'], p, 1)  # Slightly larger nodes
        pygame.draw.circle(map_surface, NODE_COLOR, p, 1)  # Slightly larger nodes
    except Exception:
        continue


# Drawing extracted labels
for label, center in label_list:
    text_surface = render_text_scaled(pygame.font.get_default_font(), label, 16, HEADING_TEXT_COLOR)
    text_rect = text_surface.get_rect(center=(int(center[0]), int(center[1])))
    map_surface.blit(text_surface, text_rect)


class SquareButton:
    def __init__(self, x, y, height, text, font_size, text_color, hover_text_color, background_color, border_color, border_width):
        self.rect = pygame.Rect(int(x), int(y), int(height * 8), int(height))
        self.text = text
        self.font_size = font_size
        self.text_color = text_color
        self.hover_text_color = hover_text_color
        self.background_color = background_color
        self.border_color = border_color
        self.border_width = border_width
        self.hover_font_size = int(font_size * 1.2)
        self.is_hovered = False
        # Fonts
        self.font = pygame.font.get_default_font()
        self.hover_font = pygame.font.get_default_font()

    def draw(self, screen):
        # Create a surface with per-pixel alpha
        s = pygame.Surface((self.rect.width, self.rect.height), pygame.SRCALPHA)
        # Fill with the background color and draw rounded corners
        pygame.draw.rect(s, self.background_color, s.get_rect(), border_radius=12)
        # Blit the surface onto the main screen
        screen.blit(s, self.rect.topleft)
        # Draw the border with rounded corners
        pygame.draw.rect(screen, self.border_color, self.rect, self.border_width, border_radius=12)

        # Draw the text
        if self.is_hovered:
            text_surface = render_text_scaled(self.hover_font, self.text, self.hover_font_size, self.hover_text_color)
        else:
            text_surface = render_text_scaled(self.font, self.text, self.font_size, self.text_color)
        text_rect = text_surface.get_rect(center=self.rect.center)
        screen.blit(text_surface, text_rect)

    def update(self, mouse_pos):
        self.is_hovered = self.rect.collidepoint(mouse_pos)

    def check_click(self, mouse_pos, mouse_pressed):
        if self.rect.collidepoint(mouse_pos) and mouse_pressed:
            change_algorithm(self.text)


class ButtonLayout:
    def __init__(self, x, y, button_size, margin, rows, cols, button_texts, font_size, text_color, hover_text_color, background_color, border_color, border_width):
        self.buttons = []
        for row in range(rows):
            for col in range(cols):
                button_x = x + (button_size + margin) * col
                button_y = y + (button_size + margin) * row
                text_index = row * cols + col
                text = button_texts[text_index] if text_index < len(button_texts) else ''
                button = SquareButton(button_x, button_y, button_size, text, font_size, text_color, hover_text_color, background_color, border_color, border_width)
                self.buttons.append(button)

    def draw(self, screen):
        for button in self.buttons:
            button.draw(screen)

    def update(self, mouse_pos):
        for button in self.buttons:
            button.update(mouse_pos)

    def check_click(self, mouse_pos, mouse_pressed):
        for button in self.buttons:
            button.check_click(mouse_pos, mouse_pressed)


button_texts = ["Dijkstra's", "GBFS (Greedy BFS)", "A* (A-Star)", "BFS (Breadth First Search)", "DFS (Depth First Search)", "Bellman-Ford"]
button_layout = ButtonLayout(screen_width / 7, screen_width / 5, 50, 20, 6, 1, button_texts, 24, (255, 255, 255), (228, 177, 240, 255), BUTTON_BG_COLOR, ROAD_COLOR, 4)


def load_gif(gif_filename, target_size):
    frames = []
    try:
        with Image.open(gif_filename) as img:
            for frame in range(0, getattr(img, "n_frames", 1)):
                img.seek(frame)
                frame_img = img.convert("RGBA").copy().resize(target_size, Image.LANCZOS)
                mode = frame_img.mode
                size = frame_img.size
                data = frame_img.tobytes()
                frame_surface = pygame.image.fromstring(data, size, mode)
                frames.append(frame_surface)
    except Exception:
        # return a single placeholder frame if gif missing or fails
        s = pygame.Surface(target_size, pygame.SRCALPHA)
        pygame.draw.rect(s, (200, 200, 200), s.get_rect())
        frames = [s]
    # your original comment removed first frame; keep full list but it's safe
    return frames


def menu_screen():
    global running, last_update, frame_index

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    mouse_pos = pygame.mouse.get_pos()
    mouse_pressed = pygame.mouse.get_pressed()[0]

    screen.blit(menu_screen_bg_image, (0, 0))
    screen.blit(ribbon_label_image, (screen_width / 2 - ribbon_label_image.get_size()[0] / 2, 95))

    heading = vingue_rg_font.render("Bengaluru", True, HEADING_TEXT_COLOR)
    heading_rect = heading.get_rect()
    heading_rect.center = (screen_width / 2, 50)
    screen.blit(heading, heading_rect)

    sub_heading = calgary_font.render("Visualizing Map-Finding Algorithms - A Comparison", True, LABEL_COLOR)
    sub_heading_rect = sub_heading.get_rect()
    sub_heading_rect.center = (screen_width / 2, 100 + 35 / 2)
    screen.blit(sub_heading, sub_heading_rect)

    # Draw button
    button_layout.update(mouse_pos)
    button_layout.check_click(mouse_pos, mouse_pressed)
    button_layout.draw(screen)

    # Switching cursor icon to hand-pointer on button-hover
    any_hover = any(button.is_hovered for button in button_layout.buttons)
    try:
        if any_hover:
            pygame.mouse.set_cursor(pygame.cursors.Cursor(HAND_CURSOR))
        else:
            pygame.mouse.set_cursor(pygame.cursors.Cursor(ARROW_CURSOR))
    except Exception:
        # fallback: ignore cursor change if environment doesn't support
        pass

    current_time = pygame.time.get_ticks()
    if current_time - last_update > frame_duration:
        frame_index = (frame_index + 1) % num_frames
        last_update = current_time

    # Make sure we have at least one frame
    if num_frames > 0:
        screen.blit(frames[frame_index], (int(screen_width / 1.7), int(screen_height / 3)))


def draw_grid():
    screen.blit(map_surface, (0, 0))


def draw_path(path, color):
    global vehicle_path_updated, vehicle_path, current_vehicle_pos_index

    for i in range(len(path) - 1):
        u, v = path[i], path[i + 1]

        # Check for curved road geometry
        edge_data = G.get_edge_data(u, v)
        if not edge_data:
            # fallback to straight line if no edge data
            try:
                pygame.draw.line(screen, color, (int(pos[u][0]), int(pos[u][1])), (int(pos[v][0]), int(pos[v][1])), 3)
                if not vehicle_path_updated:
                    vehicle_path.extend(get_interpolated_positions((int(pos[u][0]), int(pos[u][1])), (int(pos[v][0]), int(pos[v][1]))))
            except Exception:
                continue
            continue

        for key in edge_data:
            road_geometry = edge_data[key].get('geometry', None)
            if road_geometry:
                road_coords = [(normalize(x, minx, maxx, 0, screen_width), normalize(y, miny, maxy, 0, screen_height)) for x, y in road_geometry.coords]
                if len(road_coords) > 1:
                    int_coords = [(int(x), int(y)) for x, y in road_coords]
                    pygame.draw.lines(screen, color, False, int_coords, 3)
                if not vehicle_path_updated:
                    for j in range(len(road_coords) - 1):
                        vehicle_path.extend(get_interpolated_positions(road_coords[j], road_coords[j + 1]))
            else:
                pygame.draw.line(screen, color, (int(pos[u][0]), int(pos[u][1])), (int(pos[v][0]), int(pos[v][1])), 3)
                if not vehicle_path_updated:
                    start_pos = (int(pos[u][0]), int(pos[u][1]))
                    end_pos = (int(pos[v][0]), int(pos[v][1]))
                    vehicle_path.extend(get_interpolated_positions(start_pos, end_pos))
    vehicle_path_updated = True


def get_interpolated_positions(start, end, steps=10):
    """Generate positions between the start and end points."""
    sx, sy = start[0], start[1]
    ex, ey = end[0], end[1]
    return [((sx + (ex - sx) * i / steps), (sy + (ey - sy) * i / steps)) for i in range(steps + 1)]


def calculate_direction(x1, y1, x2, y2):
    global directions
    angle = math.degrees(math.atan2(y2 - y1, x2 - x1))
    if angle < 0:
        angle += 360
    # find nearest existing direction (the keys in vehicle_images)
    closest_angle = min(directions, key=lambda x: abs(x - angle))
    return closest_angle


def clear_trace():
    screen.blit(map_surface, (0, 0))


def reset_trace():
    global traces, trace_complete, path, compare_paths_enabled
    # Reset traces for new start point
    traces = {
        'dijkstra': trace_dijkstra(G, start, end),
        'greedy_bfs': trace_greedy_bfs(start, end, G),
        'a_star': trace_a_star(G, start, end),
        'bellman_ford': trace_bellman_ford(G, start, end),
        'bfs': trace_bfs(G, start, end),
        'dfs': trace_dfs(G, start, end)
    }
    trace_complete = False
    path = []
    compare_paths_enabled = False


# Reverse normalize coordinates - single definition
def reverse_normalize(val, min_val, max_val, new_min, new_max):
    if (new_max - new_min) == 0:
        return (min_val + max_val) / 2
    return (val - new_min) / (new_max - new_min) * (max_val - min_val) + min_val


# A single calculate_total_distance that uses lat/lon approximations for screen scaling
def calculate_total_distance(path, lat_distance, lon_distance, screen_width, screen_height):
    total_distance = 0.0
    lat_factor = lat_distance / screen_height if screen_height != 0 else 0
    lon_factor = lon_distance / screen_width if screen_width != 0 else 0

    for u, v in zip(path[:-1], path[1:]):
        # Calculate the Euclidean distance in screen coordinates (pixel distance)
        sx, sy = pos[u]
        ex, ey = pos[v]
        screen_distance = LineString([(sx, sy), (ex, ey)]).length
        # Adjust distance using normalization factors (average)
        total_distance += screen_distance * (lat_factor + lon_factor) / 2 if (lat_factor + lon_factor) != 0 else screen_distance

    return total_distance


def compare_paths(screen, G, start, end):
    # Define your bounding box coordinates (explicit to compute lat/lon meters)
    min_lat, max_lat = 13.090, 13.150  # South, North
    min_lon, max_lon = 77.520, 77.580  # West, East

    # Calculate distances in meters for the bounding box
    lat_distance = geopy_distance((min_lat, min_lon), (max_lat, min_lon)).meters
    lon_distance = geopy_distance((min_lat, min_lon), (min_lat, max_lon)).meters

    # Initialize path dictionaries and stats
    paths = {
        'dijkstra': [],
        'greedy_bfs': [],
        'a_star': [],
        'bellman_ford': [],
        'bfs': [],
        'dfs': []
    }

    stats = {k: {'distance': 0, 'time': 0} for k in paths.keys()}

    # Calculate paths for each algorithm
    traces_local = {
        'dijkstra': trace_dijkstra(G, start, end),
        'greedy_bfs': trace_greedy_bfs(start, end, G),
        'a_star': trace_a_star(G, start, end),
        'bellman_ford': trace_bellman_ford(G, start, end),
        'bfs': trace_bfs(G, start, end),
        'dfs': trace_dfs(G, start, end)
    }

    for algo in traces_local:
        trace_obj = traces_local[algo]
        trace_complete_local = False
        start_time = pygame.time.get_ticks()

        try:
            # If the trace is a tuple (already completed), unpack it directly
            if isinstance(trace_obj, tuple):
                visited, predecessors = trace_obj
                if end in visited:
                    path_local = []
                    current = end
                    while current is not None:
                        path_local.append(current)
                        current = predecessors[current]
                    path_local.reverse()
                    paths[algo] = path_local
                    trace_complete_local = True
            else:
                # If the trace is a generator, iterate until end is found
                while not trace_complete_local:
                    visited, predecessors = next(trace_obj)
                    if end in visited:
                        path_local = []
                        current = end
                        while current is not None:
                            path_local.append(current)
                            current = predecessors[current]
                        path_local.reverse()
                        paths[algo] = path_local
                        trace_complete_local = True
        except StopIteration:
            # Generator finished without reaching end
            pass
        except Exception as e:
            print(f"Error in algorithm {algo}: {e}")
            paths[algo] = []

        end_time = pygame.time.get_ticks()
        time_elapsed = end_time - start_time  # milliseconds

        total_distance = calculate_total_distance(
            paths[algo], lat_distance, lon_distance, screen_width, screen_height
        )
        stats[algo] = {'distance': total_distance, 'time': time_elapsed}

    return paths, stats


def change_algorithm(name):
    global algorithm, current_page, trace_complete

    reset_trace()
    current_page = "visualizer"

    if name == "Dijkstra's":
        algorithm = "dijkstra"
    elif name == "GBFS (Greedy BFS)":
        algorithm = "greedy_bfs"
    elif name == "A* (A-Star)":
        algorithm = "a_star"
    elif name == "BFS (Breadth First Search)":
        algorithm = "bfs"
    elif name == "DFS (Depth First Search)":
        algorithm = "dfs"
    elif name == "Bellman-Ford":
        algorithm = "bellman_ford"


def display_legend(screen, algorithm_colors, active_algorithm, stats_dict):
    global BUTTON_BG_COLOR

    # Create a surface with per-pixel alpha
    legend_surface_width = 350
    legend_surface_height = 25 * len(algorithm_colors) + 20
    legend_surface = pygame.Surface((legend_surface_width, legend_surface_height), pygame.SRCALPHA)

    # Fill with the background color and draw rounded corners
    border_width = 2
    pygame.draw.rect(legend_surface, (228, 177, 240, 128 + 64), legend_surface.get_rect(), border_radius=6)

    # Draw the border with rounded corners
    pygame.draw.rect(legend_surface, HEADING_TEXT_COLOR, legend_surface.get_rect(), border_width, border_radius=6)

    # Font for the legend (similar to square_buttons font)
    font_inactive = pygame.font.SysFont(None, 24)
    font_active = pygame.font.SysFont(None, 28)

    # Draw each algorithm and its color on the legend
    y_offset = 10
    for algorithm, color in algorithm_colors.items():
        distance = stats_dict.get(algorithm, {}).get('distance', 0)
        time_elapsed = stats_dict.get(algorithm, {}).get('time', 0)
        try:
            dist_text = f"{distance/1000:.2f} km" if distance > 1000 else f"{distance:.2f} m"
        except Exception:
            dist_text = "N/A"
        try:
            time_text = f"{time_elapsed/60000:.2f} s" if time_elapsed > 60000 else f"{time_elapsed:.0f} ms"
        except Exception:
            time_text = "N/A"

        if algorithm == active_algorithm:
            pygame.draw.rect(legend_surface, color, (10, y_offset, 20, 20), border_radius=25)
            text_surface = font_active.render(algorithm.replace('_', ' ').title(), True, HEADING_TEXT_COLOR)
            dist_surface = font_inactive.render(dist_text, True, HEADING_TEXT_COLOR)
            time_surface = font_inactive.render(time_text, True, HEADING_TEXT_COLOR)
        else:
            pygame.draw.rect(legend_surface, color, (10 + (20 // 4), y_offset + (20 // 4), 20 // 2, 20 // 2), border_radius=25)
            text_surface = font_inactive.render(algorithm.replace('_', ' ').title(), True, HEADING_TEXT_COLOR)
            dist_surface = font_inactive.render(dist_text, True, HEADING_TEXT_COLOR)
            time_surface = font_inactive.render(time_text, True, HEADING_TEXT_COLOR)

        text_rect = text_surface.get_rect(topleft=(40, y_offset))
        dist_rect = dist_surface.get_rect(topright=(legend_surface.get_width() - 10, y_offset))
        time_rect = time_surface.get_rect(topright=(legend_surface.get_width() - 10 - 80 - 10, y_offset))

        pygame.draw.rect(legend_surface, HEADING_TEXT_COLOR, (10, y_offset, 20, 20), border_width // 2, border_radius=25)
        legend_surface.blit(text_surface, text_rect)
        legend_surface.blit(dist_surface, dist_rect)
        legend_surface.blit(time_surface, time_rect)
        y_offset += 25

    # Position the legend at the bottom left of the screen
    screen_width_local, screen_height_local = screen.get_size()
    screen.blit(legend_surface, (10, screen_height_local - legend_surface_height - 10))


def display_instructions_OR_map_heading_section(screen):
    global calgary_font, BUTTON_BG_COLOR, compare_paths_enabled, algorithm

    map_heading_surface = pygame.Surface((300, 50), pygame.SRCALPHA)
    if compare_paths_enabled:
        screen.blit(ribbon_label_image_short, (screen_width / 2 - ribbon_label_image_short.get_size()[0] / 2, 20))
        map_heading_text_surface = calgary_font.render("Comparison Mode On", True, HEADING_TEXT_COLOR)
    else:
        pygame.draw.rect(map_heading_surface, BUTTON_BG_COLOR[:3] + (128 + 64,), map_heading_surface.get_rect(), border_radius=12)
        pygame.draw.rect(map_heading_surface, (0, 0, 0), map_heading_surface.get_rect(), 2, border_radius=12)
        map_heading_text_surface = calgary_font.render(algorithm.replace('_', ' ').title() + " Algorithm", True, (255, 255, 255))
    map_heading_text_rect = map_heading_text_surface.get_rect()
    map_heading_text_rect.center = (map_heading_surface.get_size()[0] // 2, map_heading_surface.get_size()[1] // 2)
    map_heading_surface.blit(map_heading_text_surface, map_heading_text_rect)

    screen.blit(map_heading_surface, (screen_width // 2 - 300 // 2, 20))

    instructions = [
        "[R] : Return to Menu",
        "[Arrow - Up/Down] : Switch Algorithm",
        "[Click a node] : Exit Comparison Mode"
    ]
    y_offset = 10
    font = pygame.font.SysFont(None, 24)
    info_section_width, info_section_height = 350, 25 * len(instructions) + 20
    info_section_surface = pygame.Surface((info_section_width, info_section_height), pygame.SRCALPHA)
    pygame.draw.rect(info_section_surface, (228, 177, 240, 128 + 64), info_section_surface.get_rect(), border_radius=6)
    pygame.draw.rect(info_section_surface, HEADING_TEXT_COLOR, info_section_surface.get_rect(), 2, border_radius=6)
    for instruction in instructions:
        info_section_text_surface = font.render(instruction, True, HEADING_TEXT_COLOR)
        info_section__text_rect = info_section_text_surface.get_rect(topleft=(10, y_offset))
        y_offset += 25
        info_section_surface.blit(info_section_text_surface, info_section__text_rect)
    if compare_paths_enabled:
        screen.blit(info_section_surface, (screen_width - info_section_surface.get_size()[0] - 10, screen_height - info_section_surface.get_size()[1] - 10))


def algo_visualizer():
    global trace_complete, running, compare_paths_enabled, traces, start, end, path, compare_paths_current_index, current_page, COLORS
    global vehicle_path_updated, vehicle_path, current_vehicle_pos_index, algorithm

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

        elif event.type == pygame.MOUSEBUTTONDOWN:
            mouse_pos = pygame.mouse.get_pos()
            clicked_buttons = pygame.mouse.get_pressed()
            for node in G.nodes():
                try:
                    node_pos = np.array(pos[node])
                    if clicked_buttons[0] and np.linalg.norm(np.array(mouse_pos) - node_pos) < 10:
                        start = node
                        vehicle_path.clear()
                        current_vehicle_pos_index = 0
                        vehicle_path_updated = False
                        reset_trace()
                        break
                    elif clicked_buttons[2] and np.linalg.norm(np.array(mouse_pos) - node_pos) < 10:
                        end = node
                        vehicle_path.clear()
                        current_vehicle_pos_index = 0
                        vehicle_path_updated = False
                        reset_trace()
                        break
                except Exception:
                    continue

        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_c:  # Press 'c' to compare paths
                compare_paths_enabled = True
                try:
                    compare_paths_current_index = list(traces.keys()).index(algorithm)
                except ValueError:
                    compare_paths_current_index = 0

            elif event.key == pygame.K_UP:
                compare_paths_current_index = (compare_paths_current_index - 1) % max(1, len(traces))
            elif event.key == pygame.K_DOWN:
                compare_paths_current_index = (compare_paths_current_index + 1) % max(1, len(traces))
            elif event.key == pygame.K_r:  # Press 'r' to return to menu screen
                current_page = "menu"

    draw_grid()

    if not trace_complete and not compare_paths_enabled:
        try:
            visited, predecessors = next(traces[algorithm])
            for node in list(visited.keys())[1:]:
                if predecessors[node] is not None:
                    u, v = predecessors[node], node
                    edge_data = G.get_edge_data(u, v)
                    if not edge_data:
                        pygame.draw.line(screen, TRACE_COLOR, (int(pos[u][0]), int(pos[u][1])),
                                         (int(pos[v][0]), int(pos[v][1])), 2)
                        continue

                    for key in edge_data:
                        road_geometry = edge_data[key].get('geometry', None)
                        if road_geometry:
                            road_coords = [(normalize(x, minx, maxx, 0, screen_width),
                                            normalize(y, miny, maxy, 0, screen_height)) for x, y in road_geometry.coords]
                            if len(road_coords) > 1:
                                pygame.draw.lines(screen, TRACE_COLOR, False,
                                                  [(int(x), int(y)) for x, y in road_coords], 2)
                        else:
                            pygame.draw.line(screen, TRACE_COLOR, (int(pos[u][0]), int(pos[u][1])),
                                             (int(pos[v][0]), int(pos[v][1])), 2)

            if end in visited:
                path_local = []
                current = end
                while current is not None:
                    path_local.append(current)
                    current = predecessors[current]
                path_local.reverse()
                path[:] = path_local
                trace_complete = True
        except StopIteration:
            pass
        except Exception:
            pass

    if trace_complete:
        clear_trace()
        draw_path(path, COLORS.get(algorithm, TRACE_COLOR))
        draw_vehicle_movement()

    if compare_paths_enabled:
        paths, stats = compare_paths(screen, G, start, end)
        path_items = list(paths.items())
        compare_paths_current_index_mod = compare_paths_current_index % max(1, len(path_items))
        path = path_items[compare_paths_current_index_mod][1]
        algo = path_items[compare_paths_current_index_mod][0] if path_items else algorithm

        clear_trace()
        draw_path(path, COLORS.get(algo, TRACE_COLOR))
        display_legend(screen, algorithm_colors=COLORS, active_algorithm=algo, stats_dict=stats)

    # Draw markers and logos
    try:
        screen.blit(fixed_marker_image, (
            int(normalize(bmsitm_coords[0], minx, maxx, 0, screen_width) - fixed_marker_image.get_width() / 2),
            int(normalize(bmsitm_coords[1], miny, maxy, 0, screen_height) - fixed_marker_image.get_height())
        ))
        screen.blit(bmsit_logo_image, (
            int(normalize(bmsitm_coords[0], minx, maxx, 0, screen_width) - bmsit_logo_image.get_width()),
            int(normalize(bmsitm_coords[1], miny, maxy, 0, screen_height) - bmsit_logo_image.get_height())
        ))
        screen.blit(home_image, (
            int(normalize(home_coords[0], minx, maxx, 0, screen_width)),
            int(normalize(home_coords[1], miny, maxy, 0, screen_height) - home_image.get_height())
        ))
        screen.blit(fixed_marker_image, (
            int(normalize(home_coords[0], minx, maxx, 0, screen_width) - fixed_marker_image.get_width() / 2),
            int(normalize(home_coords[1], miny, maxy, 0, screen_height) - fixed_marker_image.get_height())
        ))
    except Exception:
        pass

    try:
        screen.blit(start_marker_image, (int(pos[start][0] - start_marker_image.get_width() / 2),
                                         int(pos[start][1] - start_marker_image.get_height())))
        screen.blit(end_marker_image, (int(pos[end][0] - end_marker_image.get_width() / 2),
                                       int(pos[end][1] - end_marker_image.get_height())))
    except Exception:
        pass

    display_instructions_OR_map_heading_section(screen)
    algo_map = {
    "dijkstra": "Dijkstra",
    "greedy_bfs": "Greedy-BFS",
    "a_star": "A-Star",
    "bellman_ford": "Bellman-Ford",
    "bfs": "BFS",
    "dfs": "DFS"
    }
    algo_name = algo_map.get(algorithm, algorithm)
    draw_complexity_info(screen, algo_name)

def draw_vehicle_movement():
    global current_vehicle_pos_index, vehicle_path, vehicle_images
    if not vehicle_path:
        return

    current_vehicle_pos_index = max(0, current_vehicle_pos_index)

    if current_vehicle_pos_index < len(vehicle_path) - 1:
        current_pos = vehicle_path[current_vehicle_pos_index]
        next_pos = vehicle_path[current_vehicle_pos_index + 1]
        direction = calculate_direction(current_pos[0], current_pos[1], next_pos[0], next_pos[1])
        vehicle_image_to_use = vehicle_images.get(direction, list(vehicle_images.values())[0])
    else:
        current_pos = vehicle_path[min(current_vehicle_pos_index, len(vehicle_path) - 1)]
        vehicle_image_to_use = vehicle_images.get(0, list(vehicle_images.values())[0])

    try:
        screen.blit(vehicle_image_to_use,
                    (int(current_pos[0] - vehicle_image_to_use.get_width() / 2),
                     int(current_pos[1] - vehicle_image_to_use.get_height() / 2)))
    except Exception:
        pass

    current_vehicle_pos_index = (current_vehicle_pos_index + 1) % max(1, len(vehicle_path))

# Loading GIF frames
gif_filename = os.path.join("images", "map_animation.gif")
target_size = (300, 300)  # Desired size for the frames
frames = load_gif(gif_filename, target_size)
num_frames = len(frames)
frame_duration = 100  # Duration of each frame in milliseconds
frame_index = 0
last_update = pygame.time.get_ticks()


# Main loop
current_page = "menu"
algorithm = 'dijkstra'  # Set the initial algorithm to use
traces = {
    'dijkstra': trace_dijkstra(G, start, end),
    'greedy_bfs': trace_greedy_bfs(start, end, G),
    'a_star': trace_a_star(G, start, end),
    'bellman_ford': trace_bellman_ford(G, start, end),
    'bfs': trace_bfs(G, start, end),
    'dfs': trace_dfs(G, start, end)
}
trace_complete = False
path = []
clock = pygame.time.Clock()
compare_paths_enabled = False  # Flag to enable path comparison
try:
    compare_paths_current_index = list(traces.keys()).index(algorithm)
except Exception:
    compare_paths_current_index = 0

running = True
while running:
    if current_page == "menu":
        menu_screen()
    elif current_page == "visualizer":
        algo_visualizer()

    pygame.display.flip()
    clock.tick(60)

pygame.quit()