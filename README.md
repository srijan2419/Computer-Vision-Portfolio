# Computer Vision AI — Industrial Production Monitoring

**Applied computer vision • Camera-stream processing • Production event analytics**

This project is a real industrial monitoring implementation delivered through SrijanAI Innovations. It uses an RTSP camera feed and a YOLOv8-based classifier to recognize stages of a manufacturing process, turn frame-by-frame predictions into production-cycle events, and present operational data in a web dashboard.

> **Real applied engineering, not a model demo.** The emphasis is on continuous camera ingestion, an explicit state machine, persistence, recovery from interruptions, and an interface factory operators can use. Customer identity, camera access, model weights, training images, and production history have been excluded.

## 30-second overview

| | |
| --- | --- |
| **Operational problem** | Count manufacturing cycles and maintain a visual/timestamped record without relying on manual observation of a camera. |
| **What the system does** | Monitors an RTSP feed, classifies machine/drum phases, tracks production transitions, persists events, and displays shift/day reporting. |
| **Engineering ownership** | Applied vision integration, phase/event logic, Flask backend and dashboard, camera connectivity, operator workflows, and practical recovery improvements. |
| **Technologies** | **Python, Ultralytics YOLOv8, OpenCV, Flask, SQLAlchemy/SQLite, RTSP, HTML/CSS, JavaScript.** |
| **Portfolio scope** | Curated production source, **not** a runnable distribution with site credentials or proprietary model/dataset assets. |

## Explore the actual implementation

| Component | Source | Engineering detail |
| --- | --- | --- |
| Camera ingestion | [RTSP stream capture](src/backend/stream_capture.py) | Frame interval, connection management, reconnection and file output |
| Classifier and inference | [YOLO-based classifier](src/backend/tire_drum_classifier.py) | Model loading, per-image detections, batch and video inference, bounding boxes |
| Event logic | [Production monitor](src/backend/production_monitor.py) | Live inference, phase transitions, incomplete-cycle recovery, event persistence, evidence-frame selection |
| API and persistence | [Flask application](src/backend/web_app.py) | SQLAlchemy models, user/shift data, monitoring API endpoints, production log and filter settings |
| Operational recovery | [Orphan cleanup utility](src/backend/cleanup_orphans.py) · [Reindex utility](src/backend/cleanup_and_reindex.py) | Inspecting and repairing inconsistent or interrupted production state |
| Live dashboard | [Dashboard HTML](src/frontend/templates/dashboard.html) | Monitoring overview and state indicators |
| Historical analytics | [Analytics](src/frontend/templates/analytics.html) · [Production history](src/frontend/templates/production.html) | Operator-facing trends, event records and filtering |
| Evidence and operations | [Production details](src/frontend/templates/tire_detail.html) · [Settings](src/frontend/templates/settings.html) | Frame evidence, camera and threshold controls |

[**Engineering deep dive →**](ENGINEERING_NOTES.md)

## How it works

~~~mermaid
flowchart TD
    CAM["Site RTSP camera"] --> CAP["Camera frame capture / reconnect"]
    CAP --> MODEL["YOLOv8 production-phase detection"]
    MODEL --> STATE["Stateful production-cycle monitor"]
    STATE --> FILTER["Duration / duplicate-event safeguards"]
    FILTER --> DB[("SQLite / SQLAlchemy records")]
    STATE --> PHOTO["Selected production-frame evidence"]
    DB --> API["Flask APIs"]
    PHOTO --> API
    API --> UI["Live dashboard + history + analytics"]
~~~

The process-specific classes represented in the source are **empty drum**, **in process**, and **yellow cover**. A detected class is not counted as a completed production unit on its own: the monitor interprets **phase transitions and cycle duration** before writing records.

## Interview-worthy engineering challenges

1. **Detection → operational event:** Design a state machine so individual noisy frames do not automatically become completed production counts.
2. **Site reliability:** Handle intermittent RTSP connectivity, backend restarts, and existing incomplete records.
3. **False positives:** Apply a configurable minimum cycle duration rather than counting momentary misclassifications.
4. **Traceability:** Select evidence frames over a production interval and associate them with a timestamped record.
5. **Operational tools:** Support shift reporting, production history, administrative correction, and clear dashboard status.

## Important scope and rights

This repository uses **new Git history** and only **reviewed source excerpts**. No client camera addresses, device logins, personnel, sensitive site images, production database, training dataset, or YOLO weight files are included. Any setup values and client-specific identifiers were removed from representative source. The code retains imports for unshared modules, so it is **not a standalone executable build**.

Claims above reflect inspected source behavior and the project documentation; they **do not constitute independently validated accuracy, throughput, or uptime benchmarks**. The model and third-party libraries retain their own respective licenses.

**Implementation:** Srijan Choudhary / SrijanAI Innovations.  
**Distribution:** Portfolio review only; original application-specific code is not granted an open-source license.
