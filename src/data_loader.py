import os
import pandas as pd


def filter_retailrocket_events(source_csv_path, output_directory, p1_start_date="2015-08-01", p1_end_date="2015-08-15", p2_start_date="2015-08-16", p2_end_date="2015-08-20"):
    p1_start_ms = int(pd.Timestamp(f"{p1_start_date} 00:00:00").timestamp() * 1000)
    p1_end_ms = int(pd.Timestamp(f"{p1_end_date} 23:59:59.999").timestamp() * 1000)
    p2_start_ms = int(pd.Timestamp(f"{p2_start_date} 00:00:00").timestamp() * 1000)
    p2_end_ms = int(pd.Timestamp(f"{p2_end_date} 23:59:59.999").timestamp() * 1000)

    events_df = pd.read_csv(source_csv_path)

    p1_df = events_df[(events_df["timestamp"] >= p1_start_ms) & (events_df["timestamp"] <= p1_end_ms)].copy()
    p2_df = events_df[(events_df["timestamp"] >= p2_start_ms) & (events_df["timestamp"] <= p2_end_ms)].copy()

    os.makedirs(output_directory, exist_ok=True)
    p1_output_path = os.path.join(output_directory, "events_p1_training.csv")
    p2_output_path = os.path.join(output_directory, "events_p2_testing.csv")

    p1_df.to_csv(p1_output_path, index=False)
    p2_df.to_csv(p2_output_path, index=False)

    summary_metrics = {
        "period_1_records": len(p1_df),
        "period_1_visitors": int(p1_df["visitorid"].nunique()),
        "period_1_items": int(p1_df["itemid"].nunique()),
        "period_1_views": int((p1_df["event"] == "view").sum()),
        "period_1_carts": int((p1_df["event"] == "addtocart").sum()),
        "period_1_transactions": int((p1_df["event"] == "transaction").sum()),
        "period_2_records": len(p2_df),
        "period_2_visitors": int(p2_df["visitorid"].nunique()),
        "period_2_items": int(p2_df["itemid"].nunique()),
        "period_2_views": int((p2_df["event"] == "view").sum()),
        "period_2_carts": int((p2_df["event"] == "addtocart").sum()),
        "period_2_transactions": int((p2_df["event"] == "transaction").sum()),
        "overlapping_visitors": int(len(set(p1_df["visitorid"]).intersection(set(p2_df["visitorid"])))),
        "overlapping_items": int(len(set(p1_df["itemid"]).intersection(set(p2_df["itemid"]))))
    }

    return p1_output_path, p2_output_path, summary_metrics


if __name__ == "__main__":
    source_file = os.path.join("Data", "events.csv")
    target_dir = "Data"
    p1_path, p2_path, stats = filter_retailrocket_events(source_file, target_dir)
    print("Dataset filtering execution completed successfully.")
    for key, value in stats.items():
        print(f"{key}: {value}")
