# Rendering Mermaid Diagrams

The diagrams in this folder are provided as `.mmd` (Mermaid) source files.

To render them as SVG or PNG images, you can use the official Mermaid CLI.

## Installation

```bash
npm install -g @mermaid-js/mermaid-cli
```

## Usage

Render to SVG:
```bash
mmdc -i figure_1_w0035_system_pipeline.mmd -o figure_1_w0035_system_pipeline.svg -b white
mmdc -i figure_3_convnextv2_w0035_architecture.mmd -o figure_3_convnextv2_w0035_architecture.svg -b white
mmdc -i figure_8_model_development_summary.mmd -o figure_8_model_development_summary.svg -b white
```

Render to high-resolution PNG:
```bash
mmdc -i figure_1_w0035_system_pipeline.mmd -o figure_1_w0035_system_pipeline.png -b white -s 2
mmdc -i figure_3_convnextv2_w0035_architecture.mmd -o figure_3_convnextv2_w0035_architecture.png -b white -s 2
mmdc -i figure_8_model_development_summary.mmd -o figure_8_model_development_summary.png -b white -s 2
```
