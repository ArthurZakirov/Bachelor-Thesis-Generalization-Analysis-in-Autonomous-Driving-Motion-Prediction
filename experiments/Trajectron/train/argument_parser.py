"""
Arguments for Trajectron train.py
"""
import argparse


parser = argparse.ArgumentParser()

######################################
# Experiment Design
######################################

parser.add_argument(
    "--data_dir",
    help="Ordner mit Trainingsdaten. "
    "Alle im Ordner enthaltenen Files werden für das Training in einen einzigen Datensatz kombiniert. "
    "Angabe innerhalb des Ordners: --log_dir.",
    type=str,
    default="boston_middle",
)

parser.add_argument(
    "--train_data_file",
    help="Zusätzlich zum --data_dir kann ein einzelnes File ausgewählt werden. In diesem Fall werden alle anderen Files des Ordners ignoriert.",
    type=str,
    default=None,
)

parser.add_argument(
    "--eval_data_file",
    help="Zusätzlich zum --data_dir kann ein einzelnes File ausgewählt werden. In diesem Fall werden alle anderen Files des Ordners ignoriert.",
    type=str,
    default=None,
)


############################################
# Model Design
############################################

parser.add_argument(
    "--model_tag",
    help="Bezeichnung der Trajectron++ Variante. Bestimmt den Speicherort des trainierten Modells.",
    type=str,
    default="robot",
)

parser.add_argument(
    "--experiment_tag",
    help="Bezeichnung des Experiments. Bestimmt den Speicherort des trainierten Modells.",
    type=str,
    default="",
)

parser.add_argument(
    "--incl_robot_node",
    help="boolean Variable, die angibt ob Trajectron++ die Robot Funktionalität verwendet.",
    action="store_true",
)

parser.add_argument(
    "--map_encoding",
    help="boolean Variable, die angibt ob Trajectron++ die Karteninformation verwendet.",
    action="store_true",
)

parser.add_argument(
    "--edge_encoding",
    help="boolean Variable, die angibt ob Trajectron++ die Interaktion verwendet.",
    action="store_true",
)

parser.add_argument(
    "--dynamic_edges",
    help="whether to use dynamic edges or not, options are 'no' and 'yes'",
    type=str,
    default="yes",
)

parser.add_argument(
    "--edge_state_combine_method",
    help="the method to use for combining edges of the same type",
    type=str,
    default="sum",
)

parser.add_argument(
    "--edge_influence_combine_method",
    help="the method to use for combining edge influences",
    type=str,
    default="attention",
)

parser.add_argument(
    "--edge_addition_filter",
    nargs="+",
    help="what scaling to use for edges as they're created",
    type=float,
    default=[0.25, 0.5, 0.75, 1.0],
)  # We don't automatically pad left with 0.0, if you want a sharp
# and short edge addition, then you need to have a 0.0 at the
# beginning, e.g. [0.0, 1.0].

parser.add_argument(
    "--edge_removal_filter",
    nargs="+",
    help="what scaling to use for edges as they're removed",
    type=float,
    default=[1.0, 0.0],
)  # We don't automatically pad right with 0.0, if you want a sharp drop off like
# the default, then you need to have a 0.0 at the end.

parser.add_argument(
    "--override_attention_radius",
    action="append",
    help='Specify one attention radius to override. E.g. "PEDESTRIAN VEHICLE 10.0"',
    default=[],
)

parser.add_argument("--min_scene_interaction_density", type=float, default=-1)


############################################
# Data Design
############################################

parser.add_argument(
    "--ph",
    help="prediction_horizon: Anzahl der Zeitschritte in der Zukunft.",
    type=int,
    default=12,
)

parser.add_argument(
    "--max_hl",
    help="maximum_history_length: Anzahl der Zeitschritte in der Vergangenheit",
    type=int,
    default=4,
)

parser.add_argument(
    "--return_robot",
    help="Falls die int_ee_me configuration von Trajectron++ verwendet wird, sagt diese boolean Variable, ob die Samples des Ego Fahrzeugs im Training als gewöhnlicher Agent verwendet werden sollen.",
    action="store_true",
)

parser.add_argument(
    "--patch_size",
    help="Größe der Karte (in Pixel) aus Fahrzeugperspektive in folgende Richtungen [links, hinten, rechts, vorne]",
    type=int,
    nargs="+",
    default=[50, 10, 50, 90],
)

parser.add_argument(
    "--map_layers",
    help="Liste der zu verwendenden Kartenschichten: Auswahl aus 'drivable_area', 'road_divider', 'lane_divider'",
    type=str,
    nargs="+",
    default=["drivable_area"],
)

parser.add_argument(
    "--augment",
    help="boolean Variable, die angibt ob die Trainingsdaten augmentiert werden sollen.",
    action="store_true",
)

parser.add_argument(
    "--node_freq_mult_train",
    help="Whether to use frequency multiplying of nodes during training",
    action="store_true",
)

parser.add_argument(
    "--node_freq_mult_eval",
    help="Whether to use frequency multiplying of nodes during evaluation",
    action="store_true",
)

parser.add_argument(
    "--scene_freq_mult_train",
    help="Whether to use frequency multiplying of nodes during training",
    action="store_true",
)

parser.add_argument(
    "--scene_freq_mult_eval",
    help="Whether to use frequency multiplying of nodes during evaluation",
    action="store_true",
)

parser.add_argument(
    "--scene_freq_mult_viz",
    help="Whether to use frequency multiplying of nodes during evaluation",
    action="store_true",
)


######################################
# Training Design
######################################

parser.add_argument(
    "--train_epochs", help="number of iterations to train for", type=int, default=70
)

parser.add_argument("--batch_size", help="training batch size", type=int, default=256)

parser.add_argument("--lr", help="learning rate", type=float, default=0.002)

parser.add_argument("--use_lr_scheduler", action="store_true")

parser.add_argument(
    "--num_train_samples",
    help="Maximale Trainingsdatensatz Größe. Auswahl der Samples erfolgt zufällig.",
    type=int,
    default=10000,
)


parser.add_argument(
    "--percentages",
    help="Falls Datensätze zum Training kombiniert werden,"
    "soll eine Liste mit den prozentuellen Anteilen der Trainingsdaten übergeben werden."
    "Die Anteile müssen in alphabetischer Reihenfolge gelistet sein.",
    nargs="*",
    type=float,
    default=[None],
)

parser.add_argument(
    "--speed",
    help="Geschwindigkeitsbereich des Trainingsdatensatzes. Auswahl zwischen 'slow', 'middle', 'fast'",
    type=str,
    default="",
)

parser.add_argument(
    "--shuffle_loader",
    help="Ob die Samples im Trainingsdatensatz zufällig gemischt werden sollen.",
    type=bool,
    default=True,
)

parser.add_argument(
    "--seed", help="manual seed to use, default is 123", type=int, default=123
)

######################################
# Training Configuration
######################################

parser.add_argument(
    "--load_model_dir",
    help="Wenn Fortsetzung eines Trainings gewünscht gebe den Pfad zum Modell innerhalb von --log_dir an.",
    type=str,
    default=None,
)

parser.add_argument(
    "--conf",
    help="Pfad zur ursprünglichen Trajectron++ configuration. Auswahl aus: "
    "'../models/original_config/int_ee_me/config.json' "
    "oder '../models/original_config/robot/config.json'",
    type=str,
    default="../models/original_config/robot/config.json",
)

parser.add_argument(
    "--log_dir",
    help="Pfad zum übergeordneten Ordner an dem die Trainingsinformationen aller Modelle gespeichert werden.",
    type=str,
    default="../models",
)

parser.add_argument(
    "--device", help="what device to perform training on", type=str, default="cuda:0"
)

parser.add_argument(
    "--eval_device", help="what device to use during evaluation", type=str, default=None
)

parser.add_argument(
    "--debug", help="disable all disk writing processes.", action="store_true"
)

parser.add_argument(
    "--preprocess_workers",
    help="number of processes to spawn for preprocessing",
    type=int,
    default=0,
)

parser.add_argument(
    "--offline_scene_graph",
    help="whether to precompute the scene graphs offline, options are 'no' and 'yes'",
    type=bool,
    default="True",
)

parser.add_argument(
    "--save_every",
    help="how often to save during training, never if None",
    type=int,
    default=1,
)


####################################
# Evaluation
####################################

parser.add_argument(
    "--num_eval_samples",
    help="Maximale Evaluationsdatensatz Größe. Auswahl der Samples erfolgt zufällig.",
    type=int,
    default=1000,
)

parser.add_argument(
    "--eval_batch_size", help="evaluation batch size", type=int, default=256
)

parser.add_argument(
    "--k_eval", help="how many samples to take during evaluation", type=int, default=25
)

parser.add_argument(
    "--eval_every",
    help="how often to evaluate during training, never if None",
    type=int,
    default=1,
)

parser.add_argument(
    "--num_eval_scenes",
    help="Anzahl der in der Evaluation verwendeten Szenen.",
    type=int,
    default=5,
)

parser.add_argument(
    "--vis_every",
    help="how often to visualize during training, never if None",
    type=int,
    default=None,
)

args = parser.parse_args()
