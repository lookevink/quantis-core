import json

from lab.fault_matrix.summarize_contextual_jepa_ablations import (
    CANDIDATES,
    markdown_summary,
    summarize,
)


def test_ablation_summary_selects_only_from_training_family_folds(
    tmp_path,
) -> None:
    fold_rates = {
        "huber-log1": (0.03, "failed"),
        "huber-log2": (0.025, "passed"),
        "l1-log1": (0.02, "passed"),
        "mse-log1": (0.028, "failed"),
    }
    for index, name in enumerate(CANDIDATES):
        candidate = tmp_path / name
        candidate.mkdir()
        rate, status = fold_rates[name]
        (candidate / "development.json").write_text(
            json.dumps(
                {
                    "config": {
                        "loss": name.split("-")[0],
                        "log_latent_dimension": (
                            2 if name.endswith("log2") else 1
                        ),
                    },
                    "selection": {"status": status},
                    "cross_validation": {
                        "summary": {
                            "contextual_mean_alert_rate": rate,
                            "metrics_only_mean_alert_rate": 0.027,
                            "no_worse_fold_fraction": 0.75,
                        }
                    },
                    "metrics": {
                        "contextual_multimodal": {
                            "validation": {
                                "alert_rate": 0.5 - index * 0.1
                            }
                        },
                        "shuffled_logs": {
                            "validation": {"alert_rate": 0.01}
                        },
                    },
                }
            )
        )

    summary = summarize(tmp_path)

    assert summary["selected_candidate"] == "l1-log1"
    assert summary["selection_uses_exposed_validation"] is False
    assert summary["publication_eligible"] is False
    report = markdown_summary(summary)
    assert "Selected candidate: **l1-log1**" in report
    assert "new untouched corpus" in report
