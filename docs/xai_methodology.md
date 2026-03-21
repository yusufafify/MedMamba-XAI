# XAI Methodology: SSM-GradCAM

## Overview

Standard explainability methods for deep networks rely on either:

- **Attention maps** (ViTs) — directly inspecting the attention weight matrices
- **GradCAM** (CNNs) — gradient-weighted activation maps from convolutional layers

State-Space Models (SSMs) have *neither* explicit attention matrices *nor*
convolutional feature maps in the traditional sense. This document describes
**SSM-GradCAM**, our adaptation of GradCAM to the VMamba architecture.

---

## Method

### Core Insight

Although VSSBlocks use selective state-space scans rather than convolutions,
each stage still produces a **spatial feature map** of shape `(B, C, H', W')`.
These feature maps encode the learned representations at a given spatial
resolution and can be treated analogously to CNN feature maps.

### Algorithm

Given a trained model, an input image, and a target class `c`:

1. **Forward pass** — Run the image through the model, collecting the
   intermediate feature map `A_k` at stage `k` (default: last stage).

2. **Backward pass** — Compute the gradient of the target logit `y_c` with
   respect to `A_k`:

   ```
   G_k = ∂y_c / ∂A_k     ∈ ℝ^(1 × C × H' × W')
   ```

3. **Global average pooling** of gradients to obtain per-channel importance
   weights:

   ```
   w_c = (1 / (H' × W'))  Σ_{i,j}  G_k[:, :, i, j]     ∈ ℝ^(1 × C)
   ```

4. **Weighted combination** of feature map channels:

   ```
   M_c = ReLU( Σ_c  w_c · A_k[:, c, :, :] )     ∈ ℝ^(H' × W')
   ```

   The ReLU ensures we only highlight regions that *positively* contribute
   to the target class prediction.

5. **Normalisation** — Scale to `[0, 1]`:

   ```
   M_c ← M_c / max(M_c)
   ```

6. **Upsampling** — Bilinearly interpolate `M_c` from `(H', W')` to the
   original input resolution `(H, W)`.

### Why This Works for SSMs

The key observation is that the cross-scan aggregation in VSSBlocks still
*produces* spatially-structured feature maps. Even though the internal
computation routes tokens through multiple 1-D scans, the output is reshaped
back to a 2-D grid. Therefore:

- The spatial correspondence between feature map positions and input regions
  is preserved.
- Gradients flow through the SSM computation graph normally via autograd.
- Channel importance (step 3) captures which state-space channels the model
  relies on for a given prediction.

### Limitations

1. **Scan-direction artefacts** — Because the SSM processes tokens in four
   directional orders, saliency may exhibit subtle directional biases.
2. **Coarse resolution** — The last stage typically has 7×7 spatial resolution,
   which limits the spatial granularity of the heatmap.
3. **Class-conditional** — Like standard GradCAM, the map is specific to one
   target class. For untargeted importance, the predicted class is used.

---

## Multi-Stage Inspection

By varying `target_stage` from 0 to 3, one can inspect how the model's
spatial focus evolves from early (fine-grained texture) to late (semantic
region) stages. This is demonstrated in `notebooks/03_xai_demo.ipynb`.

---

## Batch Generation

For efficiency, `SSMGradCAM.generate_batch()` loops over individual images
rather than batching, because each image requires an independent backward
pass. This is a standard limitation shared by all gradient-based saliency
methods.

---

## References

- Selvaraju, R. R. et al. (2017). *Grad-CAM: Visual Explanations from Deep
  Networks via Gradient-based Localization.* ICCV.
- Liu, Y. et al. (2024). *VMamba: Visual State Space Model.* arXiv:2401.10166.
- Gu, A. & Dao, T. (2023). *Mamba: Linear-Time Sequence Modeling with
  Selective State Spaces.* arXiv:2312.00752.
