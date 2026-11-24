---

title: Solver visualizer
description: Live WASM-driven 2D optimization trajectories from the Basin Rust library.
layout: page
---

<script setup lang="ts">
import { onMounted, shallowRef } from "vue";
import type { Component } from "vue";

const visualizer = shallowRef<Component>();
onMounted(async () => {
    visualizer.value = (await import("#lib/Visualizer.vue")).default;
});
</script>

<div class="vp-raw">
  <component :is="visualizer" v-if="visualizer" />
  <section v-else class="visualizer-container py-6">
    <h1>Solver visualizer</h1>
    <p>Live optimization trajectories from the Basin Rust library.</p>
    <p>Loading visualizer…</p>
  </section>
</div>
