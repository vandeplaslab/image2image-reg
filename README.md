# image2image-reg

[![License](https://img.shields.io/pypi/l/image2image-reg.svg?color=green)](https://github.com/vandeplaslab/image2image-reg/raw/main/LICENSE)
[![PyPI](https://img.shields.io/pypi/v/image2image-reg.svg?color=green)](https://pypi.org/project/image2image-reg)
[![Python Version](https://img.shields.io/pypi/pyversions/image2image-reg.svg?color=green)](https://python.org)
[![CI](https://github.com/vandeplaslab/image2image-reg/actions/workflows/ci.yml/badge.svg)](https://github.com/vandeplaslab/image2image-reg/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/vandeplaslab/image2image-reg/branch/main/graph/badge.svg)](https://codecov.io/gh/vandeplaslab/image2image-reg)

Whole slide image registration using elastix and/or Valis-WSI.

## Overview

`image2image-reg` is a library for whole-slide image registration. It provides a command-line interface for performing
registration, transformation and evaluation of whole-slide image registrations. The library is inspired by `wsireg` and
`valis`. In fact, the registration workflow is a complete rewrite of the `wsireg` library to provide more flexibility
and control over the registration process.

## Differences from `wsireg`

- [DONE] Registrations are organized into `projects`. Project is a collection of configuration, logs, visualisations and images.
This should provide a more structured way of organizing your images and not cluttering the working directory.
- [WIP] Registrations can be initialized with `affine` matrix (to get a good starting point) - this will allow for faster
registration and hopefully better results.
- [WIP] Registrations can be focused on specific region of interest (mask) - while it's possible to do this in `wsireg`,
it's not as straightforward as it should be.
- [DONE] Better support for point and GeoJSON data.
- [DONE] Better CLI interface - more structured and easier to use.
- [DONE] Better logging - more informative and easier to debug.
- [DONE] Reusable registrations - since all registration data is contained within the project, it's much easier to re-run
certain tasks (such as export images) without having to re-run the registration process.

## Planned features:

- [WIP] Better support for `Valis` registrations.
- [WIP] Better support for masks.
- Add better measures and means of testing effectiveness of registration.
- Add better multi-processing support.
- Add 3D module for elastiX registrations.

## Getting started

The library provides a command-line interface which can be invoked:

```bash
i2reg --help
```

Will generate output like this:

```bash
Usage: i2reg [OPTIONS] COMMAND1 [ARGS]... [COMMAND2 [ARGS]...]...

  Launch registration app.

Options:
  --version      Show the version and exit.
  --dev          Flat to indicate that CLI should run in development mode and catch all errors.
  --no_color     Flag to disable colored logs (essential when logging to file).
  -q, --quiet    Minimal output - only errors and exceptions will be shown.
  --debug        Maximum output - all messages will be shown.
  -v, --verbose  Verbose output. This is additive flag so `-vvv` will print `INFO` messages and -vvvv will print
                 `DEBUG` information.
  --log FILE     Write logs to file (specify log path).
  -h, --help     Show this message and exit.

Project:
  new             Create a new project.
  about           Print information about the registration...
  validate        Validate project configuration.
  add-image       Add images to the project.
  add-path        Specify the registration path between the...
  add-attachment  Add attachment image to registered modality.
  add-points      Add attachment points (csv/tsv/txt) to...
  add-shape       Add attachment shape (GeoJSON) to...
  add-merge       Specify how (if) images should be merged.

Execute:
  preprocess  Preprocess images.
  register    Register images.
  clear       Clear project data...
  export      Export images.

Valis:
  valis-init      Initialize Valis configuration file.
  valis-register  Register images using the Valis algorithm.

Utility:
  merge    Export images.
  convert  Convert images to pyramidal OME-TIFF.
```

## Contributing

Contributions are always welcome. Please feel free to submit PRs with new features, bug fixes, or documentation improvements.

```bash
git clone https://github.com/vandeplaslab/image2image-reg.git

pip install -e .[dev]
```
