import Metal

/// Runtime-selected resident affine GEMV for the native Q4 and OptiQ Q6
/// decoder cores. The public call shape intentionally matches the old Q4
/// wrapper so model forward paths do not duplicate projection logic.
final class AffineGEMV {
    private enum Implementation {
        case int4(DequantInt4GEMV)
        case int6(DequantInt6GEMV)
        case int8(DequantInt8GEMV)
    }

    private let implementation: Implementation
    let weightBits: Int
    let groupSize: Int

    init(context: MetalContext,
         weightBits: Int,
         groupSize: Int,
         additionalShapes: [(m: Int, n: Int)] = []) throws {
        self.weightBits = weightBits
        self.groupSize = groupSize
        switch (weightBits, groupSize) {
        case (4, 64):
            implementation = .int4(try DequantInt4GEMV(
                context: context, additionalShapes: additionalShapes))
        case (6, 128):
            implementation = .int6(try DequantInt6GEMV(context: context))
        case (8, 64):
            implementation = .int8(try DequantInt8GEMV(
                context: context, additionalShapes: additionalShapes))
        default:
            throw ModelError.indexCorrupt(
                detail: "unsupported affine GEMV Q\(weightBits)/group-\(groupSize)")
        }
    }

    func encode(commandBuffer: MTLCommandBuffer,
                weights: MTLBuffer, weightsOffset: Int = 0,
                scales: MTLBuffer, scalesOffset: Int = 0,
                biases: MTLBuffer, biasesOffset: Int = 0,
                x: MTLBuffer, xOffset: Int = 0,
                y: MTLBuffer, yOffset: Int = 0,
                m: UInt32, n: UInt32,
                outputFloat32: Bool = false) {
        switch implementation {
        case .int4(let kernel):
            kernel.encode(commandBuffer: commandBuffer,
                          weights: weights, weightsOffset: weightsOffset,
                          scales: scales, scalesOffset: scalesOffset,
                          biases: biases, biasesOffset: biasesOffset,
                          x: x, xOffset: xOffset, y: y, yOffset: yOffset,
                          m: m, n: n, outputFloat32: outputFloat32)
        case .int6(let kernel):
            precondition(!outputFloat32, "Q6 FP32 output is not used by the DSV4 core")
            kernel.encode(commandBuffer: commandBuffer,
                          weights: weights, weightsOffset: weightsOffset,
                          scales: scales, scalesOffset: scalesOffset,
                          biases: biases, biasesOffset: biasesOffset,
                          x: x, xOffset: xOffset, y: y, yOffset: yOffset,
                          m: m, n: n, groupSize: UInt32(groupSize))
        case .int8(let kernel):
            precondition(!outputFloat32, "Q8 FP32 output is not used")
            kernel.encode(commandBuffer: commandBuffer,
                          weights: weights, weightsOffset: weightsOffset,
                          scales: scales, scalesOffset: scalesOffset,
                          biases: biases, biasesOffset: biasesOffset,
                          x: x, xOffset: xOffset, y: y, yOffset: yOffset,
                          m: m, n: n)
        }
    }
}
