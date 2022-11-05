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
    <div>
      <el-radio-group v-model="viewChoice" v-show="loaded">
        <el-radio label="torch" size="large"> PyTorch Code </el-radio>
        <el-radio label="tvm" size="large"> TVM Code </el-radio>
        <el-radio label="vars" size="large"> Variable Fills </el-radio>
        <el-radio label="vis" size="large"> Visualization </el-radio>
      </el-radio-group>
      <div class="code-style">
        <div v-show="viewChoice === 'torch'" v-html="torchCode"/>
        <div v-show="viewChoice === 'tvm'" v-html="tvmCode"/>
        <div v-show="viewChoice === 'vars'" v-html="varsCode"/>
      </div>
      <div v-show="viewChoice === 'vis'">
        Working In Progress
      </div>
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
      loaded: false,
      loading: false,
      torchCode: '',
      tvmCode: '',
      varsCode: '',
      viewChoice: 'torch'
    }
  },
  methods: {
    async sample() {
      if (this.loading)
        return
      this.loading = true
      fetch(`https://api.randomkernel.com/?p=${this.paramsBudget / 100.0}&f=${this.flopsBudget / 100.0}`)
          .then(response => response.json())
          .then(response => {
            this.torchCode = `${hljs.highlight('python', response['torch']).value}`
            this.tvmCode = `${hljs.highlight('python', response['tvm']).value}`
            this.varsCode = `${hljs.highlight('json', response['vars']).value}`
            this.loaded = true
            this.loading = false
          })
    }
  }
}
</script>

<style>

.code-style {
  text-align: left;
  word-break: break-all;
  word-wrap: break-word;
  white-space: pre-wrap;
  font-family: Consolas, Monaco, monospace;
}
</style>
