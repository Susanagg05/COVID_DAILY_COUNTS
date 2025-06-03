from src.pipelines.training_pipeline import training_pipeline


def test_training_runs() -> None:
    training_pipeline.run_training()
