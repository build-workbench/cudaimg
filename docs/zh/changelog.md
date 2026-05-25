# 更新日志

本页记录 Mini-OpenCV 每个版本的变更内容。

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

---

## 3.0.0 (2026-05-26)

- `openspec/` directory and all Spec-Driven Development (SDD) artifacts
- `AGENTS.md` — heavy AI instruction file with OpenSpec/SDD workflow
- `CLAUDE.md` — duplicate AI-specific configuration file
- `CONTEXT.md` — AI domain model document (not user-facing)

- `.github/copilot-instructions.md` rewritten: removed OpenSpec/SDD/BDD/RapidCheck
  references, corrected docs technology (Jekyll → VitePress), consolidated to
  accurate technical project facts only
- `CONTRIBUTING.md`: removed Spec-Driven Development Workflow section and
  obsolete resource links
- Project version bumped to 3.0.0

---

## 2.1.0 (2026-04-27)

- Comprehensive project cleanup and standardization
- Unified version number to 2.1.0 across all files
- Locked stb dependency to specific commit for reproducibility
- Fixed .gitignore conflict with .vscode/mcp.json

- specs.legacy/ directory (migrated to openspec/)
- build/ directory (build artifacts)
- Redundant changelog documentation files

---

## 2.0.0 (2026-03-24)

- Geometric transform bilinear sampling boundary coordinate zeroing
- Removed convolution dependency on global shared constant kernels
- Gaussian blur and bilateral filter sigma parameter validation
- Converged Otsu/scaling round-trip test assumptions

- Geometric and convolution regression tests
- Exception tests for invalid parameters
- Repository-level `.clang-format` file
- `ctest --output-on-failure` in CI workflow

- CUDA language standard: C++14 compatible
- Removed C++17 features blocking older NVCC versions
- Updated documentation test coverage descriptions

### Technical Notes

Focused on deterministic correctness with minimal changes. Enhanced reliability for geometric transforms, concurrent convolution, and parameter validation.

---

## 1.3.0 (2026-03-10)

- Professional landing page (`index.md`) with architecture diagram
- Complete feature matrix documenting all operators
- GPU architecture support table (Turing → Hopper)
- Technology stack documentation
- SEO-optimized `_config.yml`

- Extended README with architecture and feature tables
- Synchronized bilingual README content
- Added Jekyll exclude list for faster builds

- Badge layout issues in README.zh-CN.md

---

## 1.2.0 (2026-03-10)

- Unified `permissions: contents: read` across workflows
- Standardized concurrency configuration
- Added `actions/configure-pages@v5` to Pages workflow
- Added path filters to reduce unnecessary builds

---

## 1.1.0 (2026-03-09)

- Standardized GitHub Actions CI workflow (`.github/workflows/ci.yml`)
- Triggers: `push`, `pull_request`, `workflow_dispatch`
- CUDA container-based build validation
- `clang-format` check job

---

## 1.0.0 (2025-02-13)

- Core GPU image processing library with CUDA acceleration
- Memory management: `DeviceBuffer`, `MemoryManager`, `StreamManager`
- Processing API: `ImageProcessor`, `PipelineProcessor`

### Operators

- **Pixel**: invert, grayscale, brightness adjustment
- **Convolution**: Gaussian blur, Sobel edge, custom kernels
- **Histogram**: calculation, equalization
- **Scaling**: bilinear, nearest-neighbor interpolation
- **Morphology**: erosion, dilation, open/close/gradient/top-hat/black-hat
- **Threshold**: global, adaptive, Otsu auto
- **Color Space**: RGB/HSV/YUV conversion
- **Geometric**: rotate, flip, affine, perspective, crop, pad
- **Filters**: median, bilateral, box, sharpen, Laplacian
- **Arithmetic**: add, subtract, multiply, blend, weighted sum, abs diff
- **Pipeline**: multi-step chaining, batch async processing

### Infrastructure

- CMake build system with CUDA support
- Unit test suite (Google Test v1.14.0)
- Example programs
- MIT License
- `.editorconfig`

---

## Version Summary

| Version | Date | Highlights |
|---------|------|------------|
| [Unreleased] | — | — |
| [3.0.0] | 2026-05-26 | Remove AI frameworks, simplify repo structure |
| [2.1.0] | 2026-04-27 | Project cleanup, standardization |
| [2.0.0] | 2026-03-24 | Correctness fixes, CI hardening |
| [1.3.0] | 2026-03-10 | Pages optimization, professional docs |
| [1.2.0] | 2026-03-10 | Workflow standardization |
| [1.1.0] | 2026-03-09 | Initial CI setup |
| [1.0.0] | 2025-02-13 | Initial release |

[Unreleased]: https://github.com/LessUp/mini-opencv/compare/v3.0.0...HEAD
[3.0.0]: https://github.com/LessUp/mini-opencv/compare/v2.1.0...v3.0.0
[2.1.0]: https://github.com/LessUp/mini-opencv/compare/v2.0.0...v2.1.0
[2.0.0]: https://github.com/LessUp/mini-opencv/compare/v1.3.0...v2.0.0
[1.3.0]: https://github.com/LessUp/mini-opencv/compare/v1.2.0...v1.3.0
[1.2.0]: https://github.com/LessUp/mini-opencv/compare/v1.1.0...v1.2.0
[1.1.0]: https://github.com/LessUp/mini-opencv/compare/v1.0.0...v1.1.0
[1.0.0]: https://github.com/LessUp/mini-opencv/releases/tag/v1.0.0
