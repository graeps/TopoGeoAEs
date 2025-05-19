import json
from pathlib import Path
import os


def render_curvature_stats(json_path):
    with open(json_path, "r") as f:
        stats = json.load(f)

    html = "<h2>Curvature Error Stats</h2>"

    # Error Comparison Table
    comparisons = stats["error_comparisons"]
    errors = stats["errors"]

    html += "<h3>Error Comparisons</h3>"
    html += "<table border='1' cellspacing='0' cellpadding='5'>"
    html += "<tr><th>Comparison</th>" + "".join(f"<th>{metric}</th>" for metric in errors.keys()) + "</tr>"
    for i, comp in enumerate(comparisons):
        html += f"<tr><td>{comp}</td>" + "".join(
            f"<td>{errors[metric][i]:.4f}</td>" for metric in errors.keys()) + "</tr>"
    html += "</table><br>"

    # Curvature Std Table
    labels = stats["curvature_std"]["labels"]
    values = stats["curvature_std"]["values"]

    html += "<h3>Curvature Std Dev</h3>"
    html += "<table border='1' cellspacing='0' cellpadding='5'>"
    html += "<tr><th>Label</th><th>Std</th></tr>"
    for label, val in zip(labels, values):
        html += f"<tr><td>{label}</td><td>{val:.4f}</td></tr>"
    html += "</table><br>"

    return html


def generate_experiment_report(config):
    config_path = os.path.join(config.log_dir, "config.json")
    with open(config_path, "w") as f:
        json.dump(vars(config), f, indent=4)

    report_path = os.path.join(config.log_dir, "report.html")

    images = sorted(Path(config.log_dir).glob("*.png"))
    json_files = sorted(Path(config.log_dir).glob("*.json"))

    with open(report_path, "w") as f:
        f.write("""
        <html>
        <head>
            <title>Experiment Report</title>
            <style>
                body {
                    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                    margin: 20px;
                    background-color: #f9f9fb;
                    color: #333;
                }
                h1 {
                    font-size: 28px;
                    color: #2c3e50;
                    border-bottom: 2px solid #ccc;
                    padding-bottom: 10px;
                }
                h2 {
                    font-size: 22px;
                    color: #34495e;
                    margin-top: 30px;
                    margin-bottom: 10px;
                }
                h3 {
                    font-size: 18px;
                    color: #2d3436;
                    margin-top: 20px;
                    margin-bottom: 8px;
                }
                table {
                    border-collapse: collapse;
                    width: 100%;
                    margin-bottom: 30px;
                    box-shadow: 0 2px 8px rgba(0,0,0,0.05);
                    border-radius: 8px;
                    overflow: hidden;
                }
                th, td {
                    padding: 12px 16px;
                    text-align: left;
                    border-bottom: 1px solid #eee;
                }
                th {
                    background-color: #f4f6f8;
                    font-weight: 600;
                    color: #2c3e50;
                }
                td {
                    background-color: #fff;
                }
                tr:hover td {
                    background-color: #f1f1f1;
                }
                img {
                    width: 95%;
                    max-width: 900px;
                    margin-bottom: 30px;
                    border-radius: 10px;
                    box-shadow: 0 2px 10px rgba(0,0,0,0.08);
                }
                .description {
                    font-size: 16px;
                    margin-bottom: 20px;
                    line-height: 1.6;
                }
            </style>
        </head>
        <body>
        """)

        f.write(f"<h1>Report: {config.experiment}</h1>")
        f.write(f"<p class='description'><b>Description:</b> {config.description}</p>")

        f.write("<h2>Plots</h2>")
        for img_path in images:
            f.write(f"<h3>{img_path.name}</h3>")
            f.write(f'<img src="{img_path.name}"><br>')

        for json_path in json_files:
            if json_path.name == "curvature_error_stats.json":
                f.write(render_curvature_stats(json_path))
            else:
                f.write(f"<h2>{json_path.name}</h2>")
                with open(json_path, "r") as jf:
                    data = json.load(jf)
                    f.write("<table>")
                    for key, value in data.items():
                        f.write(f"<tr><td><b>{key}</b></td><td>{value}</td></tr>")
                    f.write("</table><br>")

        f.write("</body></html>")
