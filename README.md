# Affordance Highlighting project
## Extensions

### Utility Files
- `utilities/positional_encoding.py`: Contains the implementation for the Positional Encoding extension.
- `utilities/prompt_enricher.py`: Handles the multi-weighted prompt generation.

### Data and Images
- **Real Object Scans** (added in the `data` class):
  - `ps5.obj`
  - `tavolo.obj`
  - `borraccia.obj`
  - `auto.obj`
- **Background Images** (in the `images` folder):
  - `background.jpg`
  - `background2.jpeg`

All these resources are utilized in the `extensions.ipynb` file.

## OpenShape Integration

The `OpenShape` folder is designed to run in the [OpenShape repository](https://github.com/Colin97/OpenShape_code.git). The following changes were made:
- The `MinkowskiFCNN` class in `OpenShape/src/models/Minkowski.py` was adapted.
- The configuration file `OpenShape/src/configs/train.yaml` was modified.

### Runnable Files
- `OpenShape/src/OpenShapeHighlighter-training.py`: For training purposes.
- `OpenShape/src/OpenShapeHighlighter.py`: For the classical OpenShapeHighlighter functionality.