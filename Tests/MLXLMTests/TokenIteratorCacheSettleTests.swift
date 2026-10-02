// Copyright © 2026 Apple Inc.

import Foundation
import MLX
import MLXLLM
import Testing

@testable import MLXLMCommon

/// `TokenIterator` settles KV cache state together with the token every
/// `cacheEvalInterval` decode steps, not on every step.
struct TokenIteratorCacheSettleTests {

    /// Counts reads of `state`, which decode steps make only to settle the cache.
    private final class StateReadCountingCache: KVCacheSimple {
        var stateReads = 0

        override var state: [MLXArray] {
            get {
                stateReads += 1
                return super.state
            }
            set { super.state = newValue }
        }
    }

    @Test("decode reads cache state only on every cacheEvalInterval-th step")
    func settlesAtTheInterval() throws {
        let config = LlamaConfiguration(
            hiddenSize: 32, hiddenLayers: 2, intermediateSize: 64, attentionHeads: 4,
            rmsNormEps: 0.00001, vocabularySize: 100, kvHeads: 2)
        let caches = [StateReadCountingCache(), StateReadCountingCache()]
        var iterator = try TokenIterator(
            input: LMInput(tokens: MLXArray([Int32(1), 2, 3])), model: LlamaModel(config),
            cache: caches, parameters: GenerateParameters(temperature: 0))

        let interval = TokenIterator.cacheEvalInterval
        var settledSteps = [Int]()
        var reads = caches[0].stateReads
        for step in 1 ... (2 * interval + 20) {
            _ = iterator.next()
            if caches[0].stateReads != reads {
                settledSteps.append(step)
                reads = caches[0].stateReads
            }
        }

        #expect(settledSteps == [interval, 2 * interval])
    }
}
