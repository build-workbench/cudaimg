# 更新日志

本项目所有值得注意的变更均记录在此文件中。

格式基于 [Keep a Changelog](https://keepachangelog.com/en/1.1.0/)，
并遵循 [语义化版本](https://semver.org/spec/v2.0.0.html)。

> **版本历史说明**：本库起源于 2025-12-31 的一次初始提交。下方 3.0.0 之前的
> 版本边界与日期是对早期工作的近似重建，并非逐一打 tag 的发布——仅 `v2.0.1`
> 与 `v2.0.2` 存在对应 git tag。因此版本标题为纯文本，而非指向发布 tag 的链接。

## Unreleased

### Removed

- `README.zh-CN.md` 与 `docs/en/` 全部英文文档，文档仅保留中文
- `.claude/` OpenSpec command and skill definitions
- GitHub Pages changelog pages under `docs/en/` and `docs/zh/`
- `docs/scripts/sync-changelog.mjs` and docs build-time changelog sync

### Changed

- `README.md` 直接采用中文内容，不再保留英文版与语言切换
- `docs/.vitepress/config.ts` 改为单语言（zh-CN），移除 en locale 与语言切换
- 工程模板中文化：PR 模板、Issue 模板、SECURITY、CONTRIBUTING、copilot-instructions
- `copilot-instructions.md` 修正事实错误（移除虚构的 30-50x 加速比、CUDA 14、
  Google Benchmark v1.8.3、双语文档等表述）
- `docs/package.json` builds docs directly without changelog sync
- `.github/workflows/docs-pages.yml` no longer rebuilds docs on root
  `CHANGELOG.md` edits alone

---

## 3.0.0 - 2026-05-26

### Removed

- `openspec/` directory and all Spec-Driven Development (SDD) artifacts
- `AGENTS.md` — heavy AI instruction file with OpenSpec/SDD workflow
- `CLAUDE.md` — duplicate AI-specific configuration file
- `CONTEXT.md` — AI domain model document (not user-facing)

### Changed

- `.github/copilot-instructions.md` rewritten: removed OpenSpec/SDD/BDD/RapidCheck
  references, corrected docs technology (Jekyll → VitePress), consolidated to
  accurate technical project facts only
- `CONTRIBUTING.md`: removed Spec-Driven Development Workflow section and
  obsolete resource links
- Project version bumped to 3.0.0

---

## 2.1.0 - 2026-04-27

### Changed

- Comprehensive project cleanup and standardization
- Unified version number to 2.1.0 across all files
- Locked stb dependency to specific commit for reproducibility
- Fixed .gitignore conflict with .vscode/mcp.json

### Removed

- specs.legacy/ directory (migrated to openspec/)
- build/ directory (build artifacts)
- Redundant changelog documentation files

---

## 2.0.0 - 2026-03-24

### Fixed

- Geometric transform bilinear sampling boundary coordinate zeroing
- Removed convolution dependency on global shared constant kernels
- Gaussian blur and bilateral filter sigma parameter validation
- Converged Otsu/scaling round-trip test assumptions

### Added

- Geometric and convolution regression tests
- Exception tests for invalid parameters
- Repository-level `.clang-format` file
- `ctest --output-on-failure` in CI workflow

### Changed

- CUDA language standard: C++14 compatible
- Removed C++17 features blocking older NVCC versions
- Updated documentation test coverage descriptions

### Technical Notes

Focused on deterministic correctness with minimal changes. Enhanced reliability for geometric transforms, concurrent convolution, and parameter validation.

---

## 1.3.0 - 2026-03-10

### Added

- Professional landing page (`index.md`) with architecture diagram
- Complete feature matrix documenting all operators
- GPU architecture support table (Turing → Hopper)
- Technology stack documentation
- SEO-optimized `_config.yml`

### Changed

- Extended README with architecture and feature tables
- Synchronized bilingual README content
- Added Jekyll exclude list for faster builds

### Fixed

- Badge layout issues in README.zh-CN.md

---

## 1.2.0 - 2026-03-10

### Changed

- Unified `permissions: contents: read` across workflows
- Standardized concurrency configuration
- Added `actions/configure-pages@v5` to Pages workflow
- Added path filters to reduce unnecessary builds

---

## 1.1.0 - 2026-03-09

### Added

- Standardized GitHub Actions CI workflow (`.github/workflows/ci.yml`)
- Triggers: `push`, `pull_request`, `workflow_dispatch`
- CUDA container-based build validation
- `clang-format` check job

---

## 1.0.0 - 2025-12-31

### Added

- Core GPU image processing library with CUDA acceleration
- Memory management: `DeviceBuffer`, `MemoryManager`
- Processing API: `ImageProcessor`, `PipelineProcessor`

### Operators

- **Pixel**: invert, grayscale, brightness adjustment
- **Convolution**: Gaussian blur, Sobel edge, custom kernels
- **Histogram**: calculation, equalization
- **Scaling**: bilinear, nearest-neighbor interpolation
- **Morphology**: erosion, dilation, open/close/gradient
- **Threshold**: global, adaptive, Otsu auto
- **Color Space**: RGB/HSV/YUV conversion
- **Geometric**: rotate, flip, affine, perspective
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
| Unreleased | — | Docs cleanup, remove AI-framework artifacts |
| 3.0.0 | 2026-05-26 | Remove AI frameworks, simplify repo structure |
| 2.1.0 | 2026-04-27 | Project cleanup, standardization |
| 2.0.0 | 2026-03-24 | Correctness fixes, CI hardening |
| 1.3.0 | 2026-03-10 | Pages optimization, professional docs |
| 1.2.0 | 2026-03-10 | Workflow standardization |
| 1.1.0 | 2026-03-09 | Initial CI setup |
| 1.0.0 | 2025-12-31 | Initial release |
