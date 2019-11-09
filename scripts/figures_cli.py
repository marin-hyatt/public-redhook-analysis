import argparse
import pandas as pandas
import os.path

#Import modules for graphing
sys.path.insert(1, '../modules')
import display

parser = argparse.ArgumentParser(description="Visualize the relationship between SPL and trucks.")
    
# Dataframe of SPL, etc
parser.add_argument("dataframe", type=argparse.FileType('r'), help="A .csv file indexed by timestamp, containing" \ 
                                                                   "information like SPL, cluster assignment, and median SPL.")

parser.add_argument("--figure_type", type=str, default="final", help="The type of figure to visualize the data as"
                                                                     "(default: final")

parser.add_argument("out_name", type=str, help="The name of the file to store the figure in (e.g. example.png)")

parser.add_argument("out_path", type=str, help="The absolute path to store the figure in")

parser.add_argument("start_date", type=str, help="Date to begin plotting from, in format YYYY-MM-DD")

parser.add_argument("end_date", type=str, help="Date to stop plotting at, in format YYYY-MM-DD")

parser.add_argument("start_time", type=str, help="Time to begin plotting from, in format HH:MM:SS (military time)")

parser.add_argument("end_time", type=str, help="Time to stop plotting at, in format HH:MM:SS (military time)")

parser.add_argument("smoothing_window", type=float, help="Parameter for smoothing the current SPL, increasing it "
                                                         "smoothes the curve more")

parser.add_argument("smoothing_window_ambient", type=float, help="Parameter for smoothing the ambient SPL, increasing "
                                                                 "it smoothes the curve more")

parser.add_argument("smoothing", type=str, help="Type of smoothing; either mean, median, or gaussian")

parser.add_argument("--ds_factor", type=int, default=1, help="Downsample factor for getting the median (default: 1)")

parser.add_argument("--peak_window_size", type=int, default=3, help="Parameter for peak picking (default: 3)")

args = parser.parse_args()

#Do stuff to arguments
df = pandas.read_csv(args.dataframe)

#Slicing the dataframe based on time
df = df[args.start_date + " " + args.start_time + "-04:00":args.end_date + " " + args.end_time + "-04:00"]

#Plot the data
if args.figure_type == "final":
    #TODO: make the functions in module return plots instead of displaying them
    plot = plot_truck_clusters_final(df, args.peak_window_size, \
                                       args.smoothing_window, args.smoothing_window_ambient, args.ds_factor, args.smoothing)
#TODO: change the name
elif args.figure_type == "first":
    plot = plot_truck_clusters(df, args.peak_window_size, \
                                       args.smoothing_window, args.smoothing_window_ambient, args.ds_factor, args.smoothing)
elif args.figure_type == "median":
    plot = plot_truck_clusters_median(df, args.peak_window_size, \
                                       args.smoothing_window, args.smoothing_window_ambient, args.ds_factor, args.smoothing)
elif args.figure_type == "median_shading":
    plot = plot_truck_clusters_median_shading(df, args.peak_window_size, \
                                       args.smoothing_window, args.smoothing_window_ambient, args.ds_factor, args.smoothing)
elif args.figure_type == "normalized":
    plot = plot_truck_clusters_normalized(df, args.peak_window_size, \
                                       args.smoothing_window, args.smoothing_window_ambient, args.ds_factor, args.smoothing)
elif args.figure_type == "normalized_final":
    plot = plot_truck_clusters_normalized_final(df, args.peak_window_size, \
                                       args.smoothing_window, args.smoothing_window_ambient, args.ds_factor, args.smoothing)
#Save plot to user-specified path 
plot.savefig(os.path.join(out_path, out_name))