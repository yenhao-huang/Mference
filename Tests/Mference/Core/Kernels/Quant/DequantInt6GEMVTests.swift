import Foundation
import Metal
import Testing

@testable import Mference
import MferenceValidationSupport

@Suite struct DequantInt6GEMVTests {
    @Test func denseBitstreamGroup128MatchesScalarReference() throws {
        let m = 17
        let n = 256
        let groupSize = 128
        let groups = n / groupSize
        let wordsPerRow = n * 6 / 32
        var rng = SeedTree(0x6_128_0731).key("q6-dense-bitstream")
        var quant = [UInt32](repeating: 0, count: m * n)
        var packed = [UInt32](repeating: 0, count: m * wordsPerRow)
        for row in 0..<m {
            for col in 0..<n {
                let q = UInt32(rng.next() & 0x3f)
                quant[row * n + col] = q
                let bit = col * 6
                let word = bit / 32
                let shift = bit % 32
                packed[row * wordsPerRow + word] |= q << UInt32(shift)
                if shift > 26 {
                    packed[row * wordsPerRow + word + 1] |= q >> UInt32(32 - shift)
                }
            }
        }
        var scales = [UInt16](repeating: 0, count: m * groups)
        var biases = [UInt16](repeating: 0, count: m * groups)
        for i in scales.indices {
            scales[i] = Quantization.bf16Bits(rng.uniform(0.002, 0.012))
            biases[i] = Quantization.bf16Bits(rng.uniform(-0.2, 0.0))
        }
        let x = (0..<n).map { _ in Float16(rng.uniform(-0.25, 0.25)) }
        var reference = [Float](repeating: 0, count: m)
        for row in 0..<m {
            var sum: Float = 0
            for col in 0..<n {
                let group = col / groupSize
                let scale = Quantization.bf16ToFloat(scales[row * groups + group])
                let bias = Quantization.bf16ToFloat(biases[row * groups + group])
                sum += (Float(quant[row * n + col]) * scale + bias) * Float(x[col])
            }
            reference[row] = sum
        }

        let ctx = try MetalContext()
        let kernel = try DequantInt6GEMV(context: ctx)
        let w = try #require(ctx.device.makeBuffer(
            bytes: packed, length: packed.count * MemoryLayout<UInt32>.stride,
            options: .storageModeShared))
        let s = try #require(ctx.device.makeBuffer(
            bytes: scales, length: scales.count * MemoryLayout<UInt16>.stride,
            options: .storageModeShared))
        let b = try #require(ctx.device.makeBuffer(
            bytes: biases, length: biases.count * MemoryLayout<UInt16>.stride,
            options: .storageModeShared))
        let xb = try #require(Fp16Buffer.make(ctx.device, halves: x))
        let y = try #require(Fp16Buffer.make(ctx.device, count: m))
        let cb = try #require(ctx.queue.makeCommandBuffer())
        kernel.encode(commandBuffer: cb, weights: w, scales: s, biases: b,
                      x: xb, y: y, m: UInt32(m), n: UInt32(n),
                      groupSize: UInt32(groupSize))
        cb.commit()
        cb.waitUntilCompleted()
        #expect(cb.error == nil)
        let actual = Fp16Buffer.read(y, count: m)
        let rel = RelError.compute(actual: actual, reference: reference)
        #expect(rel < Tolerance.fp16Reduction, "Q6/group-128 rel=\(rel)")
    }
}
