import json
from pathlib import Path
import os
from types import SimpleNamespace


def _describe_experiment(overrides):
    desc_lines = []
    for k, v in overrides.items():
        if k != "experiment":
            desc_lines.append(f"{k}={v}")
    return ", ".join(desc_lines)


def generate_experiments(base_configuration, parameter_grid):
    # Ensure all lists are of equal length
    lengths = [len(v) for v in parameter_grid.values()]
    if len(set(lengths)) != 1:
        raise ValueError("All parameter lists in param_grid must have the same length for synchronized iteration.")

    n = lengths[0]
    experiments = {}

    for i in range(n):
        overrides = {k: v[i] for k, v in parameter_grid.items() if v[i] != "_"}

        base_name = base_configuration.get("experiment", "default")
        name = f"exp{i:02d}_{base_name}"  # concise ID with base experiment name

        overrides["experiment"] = name

        cfg = base_configuration.copy()
        cfg.update(overrides)
        cfg["description"] = _describe_experiment(overrides)

        if cfg.get("logging"):
            default_root_log_dir = "./results"
            if cfg.get("log_dir") is None:
                log_dir = os.path.join(default_root_log_dir, cfg["dataset_name"], f"results_{name}")
            else:
                log_dir = os.path.join(cfg["log_dir"], cfg["dataset_name"], f"results_{name}")
            os.makedirs(log_dir, exist_ok=True)
            cfg["log_dir"] = log_dir
        else: cfg["log_dir"] = None

        experiments[name] = SimpleNamespace(**cfg)

    return experiments


def render_curvature_stats(json_path):
    with open(json_path, "r") as f:
        stats = json.load(f)

    comparisons = stats["error_comparisons"]
    errors = stats["errors"]
    curvature_std = stats["curvature_std"]

    html = ""

    # Table 1: Error Comparison
    html += "<h2>Curvature Comparison</h2>"
    html += "<table>"
    html += "<tr><th>Metric</th>" + "".join(f"<th>{c}</th>" for c in comparisons) + "</tr>"

    for metric, values in errors.items():
        row = f"<tr><td><b>{metric}</b></td>" + "".join(f"<td>{v:.4f}</td>" for v in values) + "</tr>"
        html += row

    html += "</table><br>"

    # Table 2: Curvature Standard Deviations
    html += "<h2>Curvature Standard Deviations</h2>"
    html += "<table>"
    html += "<tr><th>Label</th><th>Std. Deviation</th></tr>"
    for label, value in zip(curvature_std["labels"], curvature_std["values"]):
        html += f"<tr><td>{label}</td><td>{value:.4f}</td></tr>"

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
                    font-size: 13px;
                }
                h1 {
                    font-size: 22px;
                    color: #2c3e50;
                    border-bottom: 2px solid #ccc;
                    padding-bottom: 6px;
                }
                h2 {
                    font-size: 16px;
                    color: #34495e;
                    margin-top: 24px;
                    margin-bottom: 6px;
                }
                h3 {
                    font-size: 14px;
                    color: #2d3436;
                    margin-top: 18px;
                    margin-bottom: 6px;
                }
                table {
                    border-collapse: collapse;
                    width: 100%;
                    margin-bottom: 20px;
                    box-shadow: 0 1px 4px rgba(0,0,0,0.04);
                    border-radius: 6px;
                    overflow: hidden;
                    font-size: 11px;
                }
                th, td {
                    padding: 4px 8px;
                    text-align: left;
                    border-bottom: 1px solid #e0e0e0;
                }
                th {
                    background-color: #f0f2f4;
                    font-weight: 600;
                    color: #2c3e50;
                }
                td {
                    background-color: #fff;
                }
                tr:hover td {
                    background-color: #f5f7f9;
                }
                img {
                    width: 95%;
                    max-width: 900px;
                    margin-bottom: 20px;
                    border-radius: 6px;
                    box-shadow: 0 1px 6px rgba(0,0,0,0.06);
                }
                .description {
                    font-size: 12px;
                    margin-bottom: 16px;
                    line-height: 1.5;
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
            if json_path.name == "curvature_errors_stats.json":
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