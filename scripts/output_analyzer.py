def print_feature_statistics(result):
    feature_names = list(result.keys())
    feature_frequencies = [len(result[name]) for name in feature_names]
    feature_ranks = rankdata(feature_frequencies, method='dense')
    feature_rank_dict = {name: rank for name, rank in zip(feature_names, feature_ranks)}

    # Sort features by their ranks
    sorted_features = sorted(feature_rank_dict.items(), key=lambda x: x[1])

    # Print results
    for feature, rank in sorted_features:
        values = result[feature]
        if isinstance(values[0], np.ndarray):
            values = [list(v) for v in values]
        print(f"{feature} (rank {rank}):")
        print(f"\tMin: {np.min(values)}")
        print(f"\tMax: {np.max(values)}")
        print(f"\tMean: {np.mean(values)}")
        print(f"\tMedian: {np.median(values)}")
        print(f"\tStd: {np.std(values)}")
        print("")
        
def convert_to_csv(result, filename):
    feature_names = list(result.keys())
    feature_frequencies = [len(result[name]) for name in feature_names]
    feature_ranks = rankdata(feature_frequencies, method='dense')
    feature_rank_dict = {name: rank for name, rank in zip(feature_names, feature_ranks)}

    # Sort features by their ranks
    sorted_features = sorted(feature_rank_dict.items(), key=lambda x: x[1])

    with open(filename, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Feature', 'Rank', 'Min', 'Max', 'Mean', 'Median', 'Std'])
        for feature, rank in sorted_features:
            values = result[feature]
            if isinstance(values[0], np.ndarray):
                values = [list(v) for v in values]
            writer.writerow([feature, rank, np.min(values), np.max(values), np.mean(values), np.median(values), np.std(values)])
            