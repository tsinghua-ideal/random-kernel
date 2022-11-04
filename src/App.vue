<template>
  <div class="common-layout">
    <h2> MLSys23/ Canvas Kernel Sampler Demo (Backbone: ResNet-18) </h2>
    FLOPs Budget: <el-input-number v-model="flopsBudget" :min="10" :max="200"/> % of Original ResNet-18
    <br><br>
    Params Budget: <el-input-number v-model="paramsBudget" :min="10" :max="200"/> % of Original ResNet-18
    <br><br>
    <el-button type="primary" v-loading="loading" @click=sample size="large">
      <b> {{ loading ? "Sampling" : "Click to Sample" }} </b>
    </el-button>
    <br><br>
    <div class="code-style">
      <div v-html="torchCode"/>
    </div>
  </div>
</template>

<script>

import hljs from "highlight.js";

export default {
  name: 'App',
  data() {
    return {
      flopsBudget: 30,
      paramsBudget: 30,
      loading: false,
      torchCode: ''
    }
  },
  methods: {
    async sample() {
      if (this.loading)
        return
      this.loading = true
      fetch(`http://43.128.41.29:5000/?p=${this.paramsBudget / 100.0}&f=${this.flopsBudget / 100.0}`)
          .then(response => response.json())
          .then(response => {
            let highlighted = hljs.highlight('python', response['torch'])
            this.torchCode = `${highlighted.value}`
            this.loading = false
          })
    }
  }
}
</script>

<style>
#app {
  font-family: Avenir, Helvetica, Arial, sans-serif;
  -webkit-font-smoothing: antialiased;
  -moz-osx-font-smoothing: grayscale;
  text-align: center;
  color: #2c3e50;
  margin-top: 60px;
}

.code-style {
  text-align: left;
  word-break: break-all;
  word-wrap: break-word;
  white-space: pre-wrap;
  font-family: Consolas, Monaco, monospace;
}
</style>
