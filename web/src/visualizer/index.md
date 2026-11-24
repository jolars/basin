---
title: Solver visualizer
description: Live WASM-driven 2D optimization trajectories from the Basin Rust
  library.
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

<div class="vp-doc visualizer-container pt-6">

# Solver visualizer

Live optimization trajectories from the Basin Rust library. Click the contour to
reset the starting point.

</div>

<div class="vp-raw">
  <component :is="visualizer" v-if="visualizer" />
  <p v-else class="visualizer-container py-6">Loading visualizer…</p>
</div>
