# SemanticSeg4EO — Documentation

This repository contains the ReadTheDocs documentation for the
**SemanticSeg4EO** QGIS plugin.

## Structure

```
semanticseg4eo-docs/
├── .readthedocs.yaml          ← ReadTheDocs configuration
├── docs/
│   ├── Makefile               ← Sphinx build commands
│   ├── requirements.txt       ← Sphinx dependencies
│   └── source/
│       ├── conf.py            ← Sphinx configuration
│       ├── index.rst          ← Documentation home page
│       ├── _static/           ← CSS and static assets
│       ├── _images/           ← Screenshots and diagrams (add yours here)
│       ├── getting_started/
│       │   ├── index.rst
│       │   ├── installation.rst
│       │   ├── environment_setup.rst
│       │   └── first_steps.rst
│       ├── user_guide/
│       │   ├── index.rst
│       │   ├── patch_extraction.rst
│       │   ├── model_training.rst
│       │   └── prediction.rst
│       └── reference/
│           ├── architectures.rst
│           ├── loss_functions.rst
│           ├── augmentation.rst
│           ├── parameters.rst
│           ├── troubleshooting.rst
│           ├── faq.rst
│           └── changelog.rst
```

## Building Locally

1. Install Sphinx dependencies:

```bash
pip install -r docs/requirements.txt
```

2. Build the HTML docs:

```bash
cd docs
make html
```

3. Open `docs/build/html/index.html` in your browser.

## Deploying to ReadTheDocs

1. Push this repository to GitHub.
2. Go to [readthedocs.org](https://readthedocs.org) and import your project.
3. ReadTheDocs will automatically use `.readthedocs.yaml` to build the docs.

## Adding Screenshots

Place your screenshots in `docs/source/_images/`.
See `docs/source/_images/README.md` for the full list of expected image files.

## License

Documentation is licensed under the MIT License — see the plugin LICENSE file.
