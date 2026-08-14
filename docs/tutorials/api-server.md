# LUNA Pipeline Metrics API & Dashboard

The `scripts/luna_analyze.py serve` command starts a local FastAPI server that exposes log metrics and a live Server-Sent Events (SSE) stream for tracking the LUNA pipeline. 

- **Default Endpoint:** `http://localhost:8000`
- **CORS:** Enabled (`allow_origins=["*"]`) for web dashboards (e.g., Nuxt running on `http://localhost:3000`).
- **Interactive Swagger UI:** `http://localhost:8000/docs`

---

## 1. Static Metrics Endpoint

### `GET /api/metrics`

Returns a complete snapshot of all scan band progress, nominal and real-time ETAs, and global mission projections.

#### Example Response
```json
{
  "source_log": "full_scan_v2.log",
  "log_size_mb": 452.8,
  "log_lines": 5863672,
  "is_global_scan": true,
  "nominal_throughput_nac_h": 193.5,
  "nominal_time_per_nac_s": 18.6,
  "bands_completed_count": 2,
  "bands_running_count": 1,
  "bands_pending_count": 7,
  "nacs_confirmed_done": 9431,
  "bands": [
    {
      "band_idx": 1,
      "min_lat": -89.999,
      "max_lat": -80.0,
      "nac_count": 545,
      "coverage_pct": 99.1,
      "pipeline_duration_s": 33602.0,
      "nodes_completed": 545,
      "status": "done",
      "confidence": "REAL"
    }
  ]
}
```

---

## 2. Live SSE Stream Endpoint

### `GET /api/live`

Server-Sent Events (SSE) stream pushing updates every second:

- **Content-Type:** `text/event-stream`
- **Payload format:** `data: <JSON_STRING>\n\n`

---

## 3. Nuxt 3 / Vue 3 Integration

Consume the live stream in a Nuxt 3 component using the native browser `EventSource`:

```vue
<template>
  <div class="p-6 bg-gray-900 text-white min-h-screen">
    <h1 class="text-2xl font-bold mb-4">LUNA Scan Dashboard</h1>
    
    <div class="mb-6 p-4 bg-gray-800 rounded-lg border border-cyan-500">
      <p class="text-sm text-cyan-400 font-mono">ACTIVE FRAME</p>
      <p class="text-3xl font-bold tracking-wider font-mono">{{ latestNac || 'WAITING...' }}</p>
    </div>
  </div>
</template>

<script setup>
import { ref, onMounted, onUnmounted } from 'vue'

const latestNac = ref(null)
let eventSource = null

onMounted(() => {
  eventSource = new EventSource('http://localhost:8000/api/live')
  eventSource.onmessage = (event) => {
    const payload = JSON.parse(event.data)
    latestNac.value = payload.latest_nac
  }
})

onUnmounted(() => {
  if (eventSource) eventSource.close()
})
</script>
```
