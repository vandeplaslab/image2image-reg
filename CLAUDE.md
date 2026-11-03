# image2image-reg
Registers whole-slide images with elastix or Valis, from preprocessing through transform export.

## Role in ecosystem
Registration engine for the image2image app.
Internal deps: image2image-io, koyo (imported, undeclared: image2image)
Depended on by: image2image (imported, undeclared: image2image-io, ionglow)

## Key abstractions
- `workflows/elastix.ElastixReg`, `workflows/valis.ValisReg` (base `workflows/_base.Workflow`)
- `models/` — `Modality`, `Preprocessing`, `Export`, `BoundingBox` / `Polygon`
- `elastix/registration.Registration` — parameter-model enum; `elastix/transform_sequence` (`Transform`, `TransformSequence`)
- `preprocessing/` — `Preprocessor` steps and `workflow.Workflow`
- `valis/` — feature detectors, `matcher.Matcher`, `slide_io.Image2ImageSlideReader`
- `cli/` — elastix, valis, merge commands

## Do not re-explain
When working in this repo, assume familiarity with: image2image, image2image-io, koyo
