from rich.console import Console
from rich.table import Table


def display_detection_stats(df_clean, blob_locations, simulated_amp):
    """
    Display detection statistics in a rich Table.

    Parameters
    ----------
    df_clean : pandas.DataFrame
        DataFrame with detected blobs and 'correct' column.
    blob_locations : list of tuple
        List of (row, col, sigma) tuples representing the true blob locations.
    """
    # Calculate statistics
    correctly_identified = df_clean["is_correct"].sum()
    total_detected = len(df_clean)
    total_actual = len(blob_locations)

    false_positives = total_detected - correctly_identified
    false_negatives = total_actual - correctly_identified

    precision = correctly_identified / total_detected if total_detected > 0 else 0
    recall = correctly_identified / total_actual if total_actual > 0 else 0
    f1_score = (
        2 * (precision * recall) / (precision + recall)
        if (precision + recall) > 0
        else 0
    )

    # Create a rich table
    console = Console()
    table = Table(
        title=f"Blob Detection Performance Metrics for Amplitude = {simulated_amp}"
    )

    # Add columns
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")
    table.add_column("Description", style="yellow")

    # Add rows
    table.add_row(
        "True Positives",
        f"{correctly_identified}",
        f"{correctly_identified}/{total_actual} actual blobs",
    )
    table.add_row(
        "False Positives",
        f"{false_positives}",
        "Detections that don't match any actual blob",
    )
    table.add_row(
        "False Negatives", f"{false_negatives}", "Actual blobs that weren't detected"
    )
    table.add_row(
        "Precision",
        f"{precision:.2f}",
        "TP/(TP+FP) - accuracy of positive predictions",
    )
    table.add_row(
        "Recall", f"{recall:.2f}", "TP/(TP+FN) - fraction of actual blobs detected"
    )
    table.add_row(
        "F1 Score", f"{f1_score:.2f}", "Harmonic mean of precision and recall"
    )

    # Print the table
    console.print(table)

    return {
        "true_positives": correctly_identified,
        "false_positives": false_positives,
        "false_negatives": false_negatives,
        "precision": precision,
        "recall": recall,
        "f1_score": f1_score,
    }


def compute_metrics_per_amplitude(
    dfs_total, truths_total, thresholds=[1e-3, 1e-2, 5e-2]
):
    """For each simulated amplitude (and corresponding simulation results),
    compute precision and recall at each given p-value threshold.

    Parameters
    ----------
    dfs_total : list of pd.DataFrame
        List of DataFrames for each simulated amplitude. Each DataFrame must contain:
          - "pvalue_harmonic_mean": p-value metric.
          - "is_correct": a Boolean column indicating if the detection was correct.
    truths_total : list of list of tuple
        List of true blob locations for each simulated amplitude.
    thresholds : list, optional
        List of p-value thresholds to evaluate. Default is [1e-3, 1e-2, 5e-2].

    Returns
    -------
    dict
        A dictionary mapping each threshold to a dictionary with keys:
            - "precision": list of precision values (one per amplitude)
            - "recall": list of recall values (one per amplitude)
            - "f1": Harmonic mean of precision and recall (one per amplitude)
    """
    metrics = {thr: {"precision": [], "recall": [], "f1": []} for thr in thresholds}

    # Loop over each simulated amplitude result.
    for df, truth in zip(dfs_total, truths_total):
        total_actual = len(truth)
        for thr in thresholds:
            # Filter detections based on the threshold.
            df_thr = df[df["pvalue_harmonic_mean"] < thr]
            total_detected = len(df_thr)
            tp = df_thr["is_correct"].sum()

            precision = tp / total_detected if total_detected > 0 else 0
            recall = tp / total_actual if total_actual > 0 else 0
            f1 = (
                2 * (precision * recall) / (precision + recall)
                if (precision + recall) > 0
                else 0
            )

            metrics[thr]["precision"].append(precision)
            metrics[thr]["recall"].append(recall)
            metrics[thr]["f1"].append(f1)

    return metrics
