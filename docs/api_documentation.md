# LUNA Pipeline Metrics API Documentation

The `scripts/luna_analyze.py serve` command starts a local FastAPI server that exposes log metrics and a live Server-Sent Events (SSE) stream for tracking the LUNA pipeline. 

- **Default Endpoint:** `http://localhost:8000`
- **CORS:** Enabled (`allow_origins=["*"]`) to allow cross-origin requests from frontends (e.g., Nuxt running on `http://localhost:3000`).
- **Interactive Docs:** Swagger UI is automatically available at `http://localhost:8000/docs` when the server is running.

---

## 1. Static Metrics Endpoint

### `GET /api/metrics`

Returns a complete, consolidated snapshot of all scan band progress, nominal and real-time ETAs, and global mission projections.

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
    },
    {
      "band_idx": 3,
      "min_lat": -50.0,
      "max_lat": -20.0,
      "nac_count": 17286,
      "coverage_pct": 99.1,
      "pipeline_duration_s": 0.0,
      "nodes_completed": 5168,
      "status": "running",
      "confidence": "REAL"
    }
  ],
  "running_band": {
    "band_idx": 3,
    "min_lat": -50.0,
    "max_lat": -20.0,
    "nac_count": 17286,
    "nodes_completed": 5168,
    "remaining": 12118,
    "nominal_eta_s": 225394.8,
    "measured_speed_nac_h": 152.6,
    "measured_eta_s": 285839.0
  },
  "global_projection": {
    "total_remaining_nacs": 66663,
    "est_remaining_time_s": 1240161.8,
    "grand_total_nacs": 81284,
    "grand_total_time_s": 1415286.8
  }
}
```

---

## 2. Live SSE Stream Endpoint

### `GET /api/live`

A Server-Sent Events (SSE) stream that pushes the latest processed LROC NAC ID and full metrics payload every second.

- **Content-Type:** `text/event-stream`
- **Payload format:** `data: <JSON_STRING>\n\n`

#### Event Payload Schema
```json
{
  "latest_nac": "M1412834034LC",
  "metrics": {
    "..."
  }
}
```

---

## Nuxt 3 / Vue 3 Integration Guide

To consume the live stream in a Nuxt 3 component or page, you can use the browser's native `EventSource` API inside `onMounted`.

### Example Page (`pages/dashboard.vue`)

```vue
<template>
  <div class="p-6 bg-gray-900 text-white min-h-screen">
    <h1 class="text-2xl font-bold mb-4">LUNA Scan Dashboard</h1>
    
    <!-- Active Frame Indicator -->
    <div class="mb-6 p-4 bg-gray-800 rounded-lg border border-cyan-500">
      <p class="text-sm text-cyan-400 font-mono">ACTIVE FRAME</p>
      <p class="text-3xl font-bold tracking-wider font-mono">{{ latestNac || 'WAITING...' }}</p>
    </div>

    <!-- Active Band Progress -->
    <div v-if="metrics?.running_band" class="mb-6 p-4 bg-gray-800 rounded-lg">
      <h2 class="text-lg font-semibold mb-2">
        Active Band {{ metrics.running_band.band_idx }} ({{ metrics.running_band.min_lat }}° to {{ metrics.running_band.max_lat }}°)
      </h2>
      <div class="w-full bg-gray-700 h-4 rounded-full overflow-hidden mb-2">
        <div class="bg-emerald-500 h-full transition-all duration-500" :style="{ width: progressPercent + '%' }"></div>
      </div>
      <div class="flex justify-between text-sm text-gray-400 font-mono">
        <span>Progress: {{ metrics.running_band.nodes_completed }} / {{ metrics.running_band.nac_count }} ({{ progressPercent }}%)</span>
        <span>Measured Speed: {{ metrics.running_band.measured_speed_nac_h || 'Calculating...' }} NACs/h</span>
      </div>
    </div>
  </div>
</template>

<script setup>
import { ref, computed, onMounted, onUnmounted } from 'vue'

const latestNac = ref(null)
const metrics = ref(null)
let eventSource = null

const progressPercent = computed(() => {
  if (!metrics.value?.running_band) return 0
  const rb = metrics.value.running_band
  return rb.nac_count > 0 ? Math.round((rb.nodes_completed / rb.nac_count) * 100) : 0
})

onMounted(() => {
  // Connect to the LUNA Metrics Server
  eventSource = new EventSource('http://localhost:8000/api/live')

  eventSource.onmessage = (event) => {
    try {
      const payload = JSON.parse(event.data)
      latestNac.value = payload.latest_nac
      metrics.value = payload.metrics
    } catch (err) {
      console.error('Failed to parse SSE payload:', err)
    }
  }

  eventSource.onerror = (err) => {
    console.error('SSE Connection Error:', err)
    eventSource.close()
  }
})

onUnmounted(() => {
  if (eventSource) {
    eventSource.close()
  }
})
</script>
```
