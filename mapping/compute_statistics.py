# take as argument a file geodesic_distrances.csv and output statistics about the geodesic distances in that file
import pandas as pd
import argparse

argument_parser = argparse.ArgumentParser(description='Compute statistics about geodesic distances.')
argument_parser.add_argument('--input_file', type=str, help='Path to the input CSV file containing geodesic distances.')
argument_parser.add_argument('--output_file', type=str, default='geodesic_statistics.txt', help='Path to the output file to save statistics.')
args = argument_parser.parse_args()

with open(args.input_file, 'r') as f:
    df = pd.read_csv(f)
    # columns : concept_id, concept_label, mapped_entity, mapped_entity_label, geodesic_distance
    # compute statistics about the geodesic distances
    geodesic_distances = df['geodesic_distance']
    
    mean_distance = geodesic_distances.mean()
    variance_distance = geodesic_distances.var()
    min_distance = geodesic_distances.min()
    max_distance = geodesic_distances.max()
    median_distance = geodesic_distances.median()
    
    with open(args.output_file, 'w') as out_f:
        out_f.write(f'Mean geodesic distance: {mean_distance}\n')
        out_f.write(f'Variance of geodesic distances: {variance_distance}\n')
        out_f.write(f'Minimum geodesic distance: {min_distance}\n')
        out_f.write(f'Maximum geodesic distance: {max_distance}\n')
        out_f.write(f'Median geodesic distance: {median_distance}\n')