# Design Decisions

Key architectural decisions and their rationale.

## Decision 1: RAII Memory Management

**Status**: Accepted

**Context**: GPU memory must be explicitly allocated and freed.

**Decision**: Use RAII pattern with `DeviceBuffer` class.

**Consequences**:
- ✅ No memory leaks
- ✅ Exception-safe
- ✅ Clear ownership semantics

## Decision 2: Three-Layer Architecture

**Status**: Accepted

**Context**: Need balance between simplicity and flexibility.

**Decision**: Separate into Application, Operator, and Infrastructure layers.

**Consequences**:
- ✅ Clear separation of concerns
- ✅ Easy to add new operators
- ✅ Testable components

## Decision 3: Shared Memory Tiling for Convolution

**Status**: Accepted

**Context**: Convolution is memory-bandwidth bound.

**Decision**: Use shared memory tiling to cache image data.

**Consequences**:
- ✅ Much less global-memory traffic than a naive convolution
- ⚠️ Kernel size limited to odd sizes up to 7×7 (larger or even sizes throw)

## Decision 4: Global-Memory Interpolation for Resize

**Status**: Accepted (texture memory not implemented)

**Context**: Image resize needs interpolation.

**Decision**: `ImageResizer` performs nearest-neighbor or bilinear interpolation with ordinary global-memory reads. CUDA texture memory is **not** used anywhere in the codebase.

**Consequences**:
- ✅ Simple implementation, no texture-object setup
- ⚠️ No hardware texture cache; bilinear weights are computed in the kernel
- ℹ️ Texture memory remains a possible future optimization; it is currently unimplemented

## Decision 5: Atomic Operations for Histogram

**Status**: Accepted

**Context**: Histogram requires counting across threads.

**Decision**: Use atomic operations with shared memory reduction.

**Consequences**:
- ✅ Correct parallel histogram
- ⚠️ Some atomic contention

## Decision 6: Unified Execution Model

**Status**: Accepted

**Context**: Callers need sync, async, and batch execution without managing CUDA streams by hand.

**Decision**: `ExecutionPolicy` (Sync/Async/Batch) owns the stream — Async/Batch create one with `cudaStreamCreate`; `ExecutionContext` pairs a policy with output allocation (`allocateOutput` / `ensureOutputSize` / `recycleToPool`). An earlier `StreamManager` stream-pool class was removed as dead code.

**Consequences**:
- ✅ One consistent interface across all operators
- ✅ Streams are created only when needed and destroyed with the policy (move-only)
- ⚠️ Memory pooling is disabled by default; opt in via `ImageAllocator::setPoolingEnabled` or `ImageProcessor::setMemoryPooling`

## See Also

- [Architecture Overview](./overview)
- [Memory Model](./memory-model)
- [CUDA Streams](./cuda-streams)
