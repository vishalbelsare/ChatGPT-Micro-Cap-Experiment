import io
import csv
from libb import LIBBmodel

def _rejected_orders_to_csv(rejected_orders: list[dict], write_header: bool) -> str:
    if not rejected_orders:
        return ""

    buffer = io.StringIO()
    fieldnames = list(rejected_orders[0].keys())
    writer = csv.DictWriter(buffer, fieldnames=fieldnames)

    if write_header:
        writer.writeheader()

    writer.writerows(rejected_orders)
    return buffer.getvalue()

def save_rejections(libb: LIBBmodel, rejected_orders: list[dict]) -> None:
    file_name = "ineligible_orders.csv"
    file_exists = (libb.layout.research_dir / "additional_logs" / file_name).exists()
    csv_text = _rejected_orders_to_csv(rejected_orders, write_header=not file_exists)
    libb.save_additional_log(file_name=file_name, text=csv_text, append=file_exists)