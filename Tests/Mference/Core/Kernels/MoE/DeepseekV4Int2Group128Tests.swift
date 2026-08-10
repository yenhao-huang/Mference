import Foundation
import Metal
import Testing

@testable import Mference
import MferenceValidationSupport

@Suite struct DeepseekV4Int2Group128Tests {
    @Test func routedFFNMatchesScalarReference() throws {
        let d = 128
        let f = 128
        let groupSize = 128
        let topK = MoEDeepseekV4.topK
        let scaleBits = Quantization.bf16Bits(0.0078125)
        let biasBits = Quantization.bf16Bits(-0.01171875)
        let scale = Quantization.bf16ToFloat(scaleBits)
        let bias = Quantization.bf16ToFloat(biasBits)

        struct Projection {
            let weights: [UInt8]
            let weightOffset: UInt32
            let scaleOffset: UInt32
            let biasOffset: UInt32
        }
        func appendProjection(rows: Int, cols: Int, seed: Int,
                              bytes: inout [UInt8]) -> Projection {
            let weightOffset = UInt32(bytes.count)
            for row in 0..<rows {
                var packed = [UInt8](repeating: 0, count: cols / 4)
                for col in 0..<cols {
                    let q = UInt8((row &* 17 &+ col &* 13 &+ seed) & 3)
                    packed[col / 4] |= q << UInt8((col & 3) * 2)
                }
                bytes.append(contentsOf: packed)
            }
            let weights = Array(bytes[Int(weightOffset)..<bytes.count])
            let scaleOffset = UInt32(bytes.count)
            for _ in 0..<(rows * cols / groupSize) {
                bytes.append(UInt8(truncatingIfNeeded: scaleBits))
                bytes.append(UInt8(truncatingIfNeeded: scaleBits >> 8))
            }
            let biasOffset = UInt32(bytes.count)
            for _ in 0..<(rows * cols / groupSize) {
                bytes.append(UInt8(truncatingIfNeeded: biasBits))
                bytes.append(UInt8(truncatingIfNeeded: biasBits >> 8))
            }
            return Projection(weights: weights, weightOffset: weightOffset,
                              scaleOffset: scaleOffset, biasOffset: biasOffset)
        }

        var blobs: [[UInt8]] = []
        var projections: [(gate: Projection, up: Projection, down: Projection)] = []
        for slot in 0..<topK {
            var bytes: [UInt8] = []
            let gate = appendProjection(rows: f, cols: d, seed: slot * 3,
                                        bytes: &bytes)
            let up = appendProjection(rows: f, cols: d, seed: slot * 3 + 1,
                                      bytes: &bytes)
            let down = appendProjection(rows: d, cols: f, seed: slot * 3 + 2,
                                        bytes: &bytes)
            blobs.append(bytes)
            projections.append((gate, up, down))
        }
        let first = projections[0]
        let offsets = MoEExpertOffsets(
            gateWOff: first.gate.weightOffset, gateSOff: first.gate.scaleOffset,
            gateBOff: first.gate.biasOffset, upWOff: first.up.weightOffset,
            upSOff: first.up.scaleOffset, upBOff: first.up.biasOffset,
            downWOff: first.down.weightOffset, downSOff: first.down.scaleOffset,
            downBOff: first.down.biasOffset)

        let x = (0..<d).map { Float16(Float(($0 % 19) - 9) / 64) }
        let residual = (0..<d).map { Float16(Float(($0 % 11) - 5) / 32) }
        let routeWeights = (0..<topK).map { Float16(Float($0 + 1) / 21) }
        func dot(_ packed: [UInt8], row: Int, cols: Int,
                 input: [Float16]) -> Float {
            let rowBase = row * cols / 4
            var sum: Float = 0
            for col in 0..<cols {
                let byte = packed[rowBase + col / 4]
                let q = (byte >> UInt8((col & 3) * 2)) & 3
                sum += (Float(q) * scale + bias) * Float(input[col])
            }
            return sum
        }
        var expectedActs = [[Float16]](repeating: [Float16](repeating: 0, count: f),
                                       count: topK)
        for slot in 0..<topK {
            for row in 0..<f {
                let gate = min(dot(projections[slot].gate.weights, row: row,
                                   cols: d, input: x), 10)
                let up = max(-10, min(10, dot(projections[slot].up.weights,
                                              row: row, cols: d, input: x)))
                expectedActs[slot][row] = Float16((gate / (1 + exp(-gate))) * up)
            }
        }
        var expected = [Float](repeating: 0, count: d)
        for row in 0..<d {
            var value = Float(residual[row])
            for slot in 0..<topK {
                value += Float(routeWeights[slot]) * dot(
                    projections[slot].down.weights, row: row, cols: f,
                    input: expectedActs[slot])
            }
            expected[row] = value
        }

        let context = try MetalContext()
        let moe = try MoEDeepseekV4(context: context,
                                    specializedD: UInt32(d),
                                    specializedF: UInt32(f),
                                    specializedNumExperts: 16,
                                    routeScale: 1.0,
                                    swigluLimit: 10.0)
        let routed = try blobs.map { bytes in
            try #require(context.device.makeBuffer(bytes: bytes, length: bytes.count,
                                                   options: .storageModeShared))
        }
        let argumentBuffer = moe.makeReusedRoutedArgumentBuffer(routedBlobs: routed)
        let xb = try #require(Fp16Buffer.make(context.device, halves: x))
        let acts = try #require(Fp16Buffer.make(context.device, count: topK * f))
        let route = try #require(Fp16Buffer.make(context.device, halves: routeWeights))
        let residualBuffer = try #require(Fp16Buffer.make(context.device, halves: residual))
        let output = try #require(Fp16Buffer.make(context.device, count: d))
        let commandBuffer = try #require(context.queue.makeCommandBuffer())
        moe.encodeRoutedPhase1(commandBuffer: commandBuffer,
                               routedArgBuffer: argumentBuffer,
                               routedBlobs: routed, routedOffsets: offsets,
                               x: xb, acts: acts, d: UInt32(d), f: UInt32(f),
                               gateGroupSize: UInt32(groupSize),
                               expertGroupSize: UInt32(groupSize))
        moe.encodeRoutedPhase2Reduce(commandBuffer: commandBuffer,
                                     routedArgBuffer: argumentBuffer,
                                     routedBlobs: routed, routedOffsets: offsets,
                                     acts: acts, routingWeights: route,
                                     residual: residualBuffer, y: output,
                                     d: UInt32(d), f: UInt32(f),
                                     expertGroupSize: UInt32(groupSize))
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
        #expect(commandBuffer.error == nil)

        let actualActs = Fp16Buffer.read(acts, count: topK * f)
        let flatExpectedActs = expectedActs.flatMap { $0 }.map(Float.init)
        #expect(RelError.compute(actual: actualActs, reference: flatExpectedActs) < 0.01)
        let actual = Fp16Buffer.read(output, count: d)
        #expect(RelError.compute(actual: actual, reference: expected) < 0.01)
    }
}
