# Local model weights

Place the original `human_detect.pt` detection model and `classify.pt`
classification model here. Model weights are ignored by Git.

The original checkpoint stores raw names `0`, `1`, `2`, `3`, `4`. These are
checked against `classification.model_class_names` at load time. The semantic
mapping is separate: the original code maps those indices to `crutches`,
`person`, `walking frame`, `walking stick`, `wheelchair`, in that order.
It is configured in `classification.class_names`.

If using another model, verify both the raw names and the semantic mapping in
`configs/crosswalk.example.toml` before running it. The metadata check validates
the raw order, not the correctness of the training labels.

Checkpoint metadata records Ultralytics `8.1.18` for `human_detect.pt` and
`8.1.14` for `classify.pt`. These are historical save versions, not a tested
installation recipe for modern Python/PyTorch.

There is no training pipeline, downloadable model URL or verified model license
included in this repository. The `demo` command does not need these files.
