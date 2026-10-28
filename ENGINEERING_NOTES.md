# Computer Vision AI — Engineering Deep Dive

This case study illustrates an industrial vision pipeline built around an actual manufacturing workflow. It is a technical demonstration of a selected code subset, not a complete model-weight or deployment package.

## 1. Camera ingestion under real site constraints

Start with [stream_capture.py](src/backend/stream_capture.py). The `RTSPFrameCapture` module attempts connection/reconnection, samples frames at intervals, and writes frames for downstream use. The portfolio **removes the original site-specific entry point and camera credential examples**, retaining the implementation of the capture class.

[ProductionMonitor](src/backend/production_monitor.py) combines a camera feed with YOLO inference and operational persistence. Its live entry point was removed because it embedded real camera access details.

## 2. Recognition is not the same as counting

The [classifier](src/backend/tire_drum_classifier.py) maps detections onto application-specific production phases. The [production monitor](src/backend/production_monitor.py) converts those observations into a persistent production lifecycle using a state machine, rather than simply counting every detection.

The monitored classes are `empty_drum`, `in_process`, and `yellow_tire` / yellow cover. A complete event depends on **transitions over time**, not a single inference output. Inspect the methods `_start_new_tire`, `_process_detection`, `_complete_tire`, and `_resume_incomplete_tire`.

A configurable minimum-duration setting can exclude short cycles that would otherwise inflate counts. **That is a heuristic with trade-offs**, not proof that false positives are impossible.

## 3. Recovering from interruptions

The original engineering documentation identifies failure modes such as duplicate detection logging and orphaned `in_process` records after a backend restart.

The [monitor](src/backend/production_monitor.py) includes a restart-resumption path. [Orphan cleanup](src/backend/cleanup_orphans.py) and [cleanup/reindex](src/backend/cleanup_and_reindex.py) illustrate practical administrative recovery. Their presence demonstrates mechanisms to *mitigate* those problems; this curated repository does not establish exhaustive concurrency or crash-recovery testing.

## 4. Evidence sampling across a long-running cycle

The completion logic stores selected frames from a buffered production run. The documented frame-selection approach samples the start, periodic points, the end, and optionally a near-end frame to avoid missing final production stages. See `_complete_tire` in the [monitor](src/backend/production_monitor.py).

This is a useful example of translating customer feedback into a time-aware data representation rather than recording arbitrary frames.

## 5. Application and operator UX

The [Flask backend](src/backend/web_app.py) models users, shifts, production records, logs, and settings. The [dashboard](src/frontend/templates/dashboard.html), [analytics](src/frontend/templates/analytics.html), [production list](src/frontend/templates/production.html), and [detail page](src/frontend/templates/tire_detail.html) expose the operational data separately from inference.

The original code contained default admin bootstrapping and real local network settings. These were **removed or replaced** in the portfolio excerpt. Source is **not a security-certified deployment example**; production hardening and site access control are separate responsibilities.

## 6. Scope and limitations

- Curated Python and Flask/Jinja code, plus a fresh Git history; some paths have been renamed for clarity.
- Camera credentials, internal addresses, user records, production data, model weights, training images, and third-party datasets excluded.
- Client/company name removed from public branding. Source reflects a **specific industrial production use case** rather than a general-purpose universal detector.
- No empirical claims about classification accuracy, throughput, 24/7 uptime, or installation count without independent measurements.
- This subset is not runnable without application components, required model weights, camera access, and configuration absent here.

Return to [README.md](README.md) for the overview.
