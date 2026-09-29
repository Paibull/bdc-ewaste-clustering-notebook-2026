import plotly.graph_objects as go
from plotly.subplots import make_subplots

COLORS = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#56B4E9"]


class ResultFigures:
    def acquisition(self, data):
        means = data.groupby("penanganan", as_index=False)["probe_auc"].mean()
        figure = go.Figure(go.Bar(
            x=means["penanganan"],
            y=means["probe_auc"],
            marker_color=COLORS[0],
        ))
        figure.update_layout(
            xaxis_title="Penanganan citra",
            yaxis_title="AUC probe sumber",
            yaxis_range=[0.45, 1.0],
            showlegend=False,
        )
        return figure

    def k_stability(self, choice, runs):
        mean = runs.groupby("k", as_index=False)["ari_to_medoid"].mean()
        count = runs.assign(stable=runs["ari_to_medoid"] >= 0.95).groupby("k")["stable"].mean()
        figure = make_subplots(specs=[[{"secondary_y": True}]])
        figure.add_trace(go.Scatter(
            x=mean["k"], y=mean["ari_to_medoid"], mode="lines+markers",
            name="ARI ke medoid", line={"color": COLORS[0]},
        ), secondary_y=False)
        figure.add_trace(go.Scatter(
            x=choice["k"], y=choice["consensus_silhouette_median"],
            error_y={
                "type": "data",
                "symmetric": False,
                "array": choice["bootstrap_ci95_high"] - choice["consensus_silhouette_median"],
                "arrayminus": choice["consensus_silhouette_median"] - choice["bootstrap_ci95_low"],
            },
            mode="lines+markers",
            name="Silhouette chooseR",
            line={"color": COLORS[1]},
        ), secondary_y=True)
        figure.add_trace(go.Bar(
            x=count.index, y=count.values, name="Seed dengan ARI ≥ 0,95",
            marker_color=COLORS[2], opacity=0.35,
        ), secondary_y=False)
        figure.update_layout(xaxis_title="Jumlah cluster (K)", barmode="overlay")
        figure.update_yaxes(title_text="Stabilitas antar-seed", secondary_y=False)
        figure.update_yaxes(title_text="Silhouette chooseR", secondary_y=True)
        return figure

    def transfer(self, data):
        rows = data[data["estimator"].isin(["fused_k12", "radio_k12", "fused_k14"])]
        figure = go.Figure(go.Bar(
            x=rows["estimator"],
            y=rows["strict_macro_top1"],
            error_y={
                "type": "data",
                "symmetric": False,
                "array": rows["strict_macro_top1_ci_high"] - rows["strict_macro_top1"],
                "arrayminus": rows["strict_macro_top1"] - rows["strict_macro_top1_ci_low"],
            },
            marker_color=[COLORS[0], COLORS[1], COLORS[2]],
        ))
        figure.update_layout(
            xaxis_title="Konfigurasi",
            yaxis_title="Macro Top-1",
            yaxis_range=[0.0, 1.0],
            showlegend=False,
        )
        return figure

    def material(self, data):
        rows = data[data["target"].isin(["six_materials", "five_materials"])]
        figure = go.Figure(go.Bar(
            x=rows["target"],
            y=rows["incremental_r2"],
            error_y={
                "type": "data",
                "symmetric": False,
                "array": rows["incremental_r2_ci_high"] - rows["incremental_r2"],
                "arrayminus": rows["incremental_r2"] - rows["incremental_r2_ci_low"],
            },
            marker_color=[COLORS[0], COLORS[2]],
        ))
        figure.update_layout(
            xaxis_title="Profil material",
            yaxis_title="Incremental R²",
            showlegend=False,
        )
        return figure

    def external_material(self, data):
        figure = go.Figure()
        for metric, color in (
            ("auroc", COLORS[0]),
            ("average_precision", COLORS[1]),
        ):
            figure.add_trace(go.Bar(
                x=data["material"],
                y=data[metric],
                name=metric.replace("_", " "),
                marker_color=color,
            ))
        figure.update_layout(
            xaxis_title="Material",
            yaxis_title="Skor",
            yaxis_range=[0.0, 1.0],
            barmode="group",
        )
        return figure

    def cluster_material(self, data):
        materials = [
            "plastic", "metal", "glass", "rubber", "paper_cardboard", "other_unknown"
        ]
        labels = [
            f"C{int(row.cluster):02d} {row.semantic_name}"
            for row in data.itertuples()
        ]
        values = data[[f"mean_{name}" for name in materials]].to_numpy(dtype=float)
        figure = go.Figure(go.Heatmap(
            z=values,
            x=["Plastik", "Logam", "Kaca", "Karet", "Kertas/karton", "Lainnya"],
            y=labels,
            colorscale="YlGnBu",
            colorbar={"title": "Proporsi"},
        ))
        figure.update_layout(
            xaxis_title="Material visual",
            yaxis_title="Cluster perangkat",
            yaxis={"autorange": "reversed"},
        )
        return figure
