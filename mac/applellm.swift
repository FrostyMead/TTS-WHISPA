// Murmr's bridge to Apple's on-device language model (Foundation Models, macOS 26+ with
// Apple Intelligence switched on). Used for AI clean-up, Command Mode and meeting
// summaries when it passes Murmr's checks (whispa/applellm.py); otherwise Murmr uses Ollama.
//
// Runs for as long as Murmr does, answering one JSON object per line on stdin:
//   {"id": 1, "op": "availability"}
//     -> {"id": 1, "available": true}  |  {"id": 1, "available": false, "reason": "..."}
//        reason: deviceNotEligible, appleIntelligenceNotEnabled, modelNotReady, osTooOld
//   {"id": 2, "op": "prewarm"}
//   {"id": 3, "op": "generate", "instructions": "...", "prompt": "...",
//    "temperature": 0.1, "max_tokens": 300}
//     -> {"id": 3, "text": "..."}  |  {"id": 3, "error": "guardrail|context|other", "message": "..."}
// Requests run concurrently (each in its own session); answers can come back in any order.
// Everything stays on the Mac: this is Apple's on-device model, not Private Cloud Compute.
//
// Build (Apple Silicon only, see .github/workflows/mac-launcher.yml):
//   swiftc -O -target arm64-apple-macos13.0 -Xlinker -weak_framework -Xlinker FoundationModels \
//          -o whispa/resources/murmr-applellm mac/applellm.swift
import Foundation
#if canImport(FoundationModels)
import FoundationModels
#endif

let writer = DispatchQueue(label: "murmr.out")

func send(_ obj: [String: Any]) {
    writer.async {
        guard let data = try? JSONSerialization.data(withJSONObject: obj) else { return }
        FileHandle.standardOutput.write(data + Data([0x0A]))
    }
}

func availability() -> [String: Any] {
    #if canImport(FoundationModels)
    if #available(macOS 26.0, *) {
        switch SystemLanguageModel.default.availability {
        case .available:
            return ["available": true]
        case .unavailable(let reason):
            switch reason {
            case .deviceNotEligible: return ["available": false, "reason": "deviceNotEligible"]
            case .appleIntelligenceNotEnabled: return ["available": false, "reason": "appleIntelligenceNotEnabled"]
            case .modelNotReady: return ["available": false, "reason": "modelNotReady"]
            @unknown default: return ["available": false, "reason": "unavailable"]
            }
        }
    }
    return ["available": false, "reason": "osTooOld"]
    #else
    return ["available": false, "reason": "notBuilt"]
    #endif
}

#if canImport(FoundationModels)
@available(macOS 26.0, *)
func makeModel() -> SystemLanguageModel {
    // Murmr only transforms text you wrote (tidying, rewriting, summarising): the guardrails
    // meant for that, so ordinary dictation with swearing or sensitive topics isn't refused.
    SystemLanguageModel(guardrails: .permissiveContentTransformations)
}

@available(macOS 26.0, *)
func generate(_ req: [String: Any]) async -> [String: Any] {
    let instructions = req["instructions"] as? String ?? ""
    let prompt = req["prompt"] as? String ?? ""
    let options = GenerationOptions(temperature: req["temperature"] as? Double,
                                    maximumResponseTokens: req["max_tokens"] as? Int)
    let session = LanguageModelSession(model: makeModel(), instructions: instructions)
    do {
        let response = try await session.respond(to: prompt, options: options)
        return ["text": response.content]
    } catch let error as LanguageModelSession.GenerationError {
        switch error {
        case .guardrailViolation: return ["error": "guardrail", "message": "\(error)"]
        case .exceededContextWindowSize: return ["error": "context", "message": "\(error)"]
        default: return ["error": "other", "message": "\(error)"]
        }
    } catch {
        return ["error": "other", "message": "\(error)"]
    }
}

@available(macOS 26.0, *)
func prewarm() {
    LanguageModelSession(model: makeModel()).prewarm()
}
#endif

func handle(_ req: [String: Any]) async -> [String: Any] {
    switch req["op"] as? String ?? "" {
    case "availability":
        return availability()
    case "prewarm":
        #if canImport(FoundationModels)
        if #available(macOS 26.0, *) { prewarm() }
        #endif
        return ["ok": true]
    case "generate":
        #if canImport(FoundationModels)
        if #available(macOS 26.0, *) { return await generate(req) }
        #endif
        return ["error": "unavailable", "message": "Apple's on-device model needs macOS 26"]
    default:
        return ["error": "other", "message": "unknown op"]
    }
}

setvbuf(stdout, nil, _IONBF, 0)
let inFlight = DispatchGroup()
Thread.detachNewThread {
    while let line = readLine() {  // ends when Murmr closes our stdin (or quits)
        guard let data = line.data(using: .utf8),
              let req = (try? JSONSerialization.jsonObject(with: data)) as? [String: Any] else { continue }
        let id = req["id"] ?? 0
        inFlight.enter()
        Task {
            var res = await handle(req)
            res["id"] = id
            send(res)
            inFlight.leave()
        }
    }
    inFlight.wait()  // answer what was asked before exiting
    writer.sync {}
    exit(0)
}
signal(SIGTERM) { _ in exit(0) }
dispatchMain()
